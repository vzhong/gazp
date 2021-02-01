import os
import re
import json
import tqdm
import utils
import torch
import random
import sqlite3
import converter
import argparse
import itertools
import embeddings as E
from vocab import Vocab
from collections import defaultdict, Counter
from transformers import DistilBertTokenizer
from eval_scripts import evaluation
from preprocess_nl2sql import BERT_MODEL, bad_query_replace, bad_question_replace, ValueAlignmentException, QueryBuildError, SQL_PRIMITIVES
from nltk.stem.porter import PorterStemmer

import editsql_preprocess
import editsql_postprocess


class SQLDataset:

    def __init__(self):
        pass

    @classmethod
    def align_values(cls, no_value, yes_value):
        if yes_value[-1] == ';':
            yes_value.pop()
        yes_value = '___'.join(yes_value)
        for f, t in bad_query_replace:
            yes_value = yes_value.replace(f, t)
        yes_value = yes_value.split('___')

        def find_match(no_value, i, yes_value):
            before = None if i == 0 else no_value[i-1].lower()
            after = None if i+1 == len(no_value) else no_value[i+1].lower()
            candidates = []

            for j in range(len(yes_value)):
                mybefore = None if j == 0 else yes_value[j-1].lower()
                if mybefore == before:
                    for k in range(j, len(yes_value)):
                        yk = yes_value[k].lower()
                        if yk in SQL_PRIMITIVES and yk not in {'in'}:
                            break
                        # if '_' in yk and 'mk_man' not in yk and 'pu_man' not in yk and 'xp_' not in yk or yk in {'t1', 't2', 't3', 't4'}:
                        #     break
                        myafter = None if k+1 == len(yes_value) else yes_value[k+1].lower()
                        if myafter == after:
                            candidates.append((j, k+1))
                            break
            if len(candidates) == 0:
                print(no_value)
                print(no_value[i])
                print(yes_value)
                import pdb; pdb.set_trace()
            candidates.sort(key=lambda x: x[1] - x[0])
            return candidates[0]

        values = []
        num_slots = 0
        for i, t in enumerate(no_value):
            t = t.lower()
            if 'value' in t and t not in {'attribute_value', 'market_value', 'value_points', 'market_value_in_billion', 'market_value_billion', 'product_characteristic_value', 'total_value_purchased'}:
                start, end = find_match(no_value, i, yes_value)
                values.append(yes_value[start:end])
                num_slots += 1
        if num_slots != len(values):
            raise Exception('Found {} values for {} slots'.format(len(values),  num_slots))
        return values

    @classmethod
    def execute(cls, db, p_str, p_sql, remap=True):
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            with utils.timeout(seconds=5, error_message='Timeout: {}'.format(p_str)):
                cursor.execute(p_str)
                p_res = cursor.fetchall()
            def res_map(res, val_units):
                rmap = {}
                for idx, val_unit in enumerate(val_units):
                    key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
                    rmap[key] = [r[idx] for r in res]
                return rmap
            if remap:
                p_val_units = [unit[1] for unit in p_sql['select'][1]]
                return res_map(p_res, p_val_units)
            else:
                return p_res
        except Exception as e:
            return []

    @classmethod
    def build_sql(cls, schema, p_str, kmap):
        try:
            p_sql = evaluation.get_sql(schema, p_str)
        except Exception as e:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = evaluation.EMPTY_QUERY.copy()
        p_valid_col_units = evaluation.build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql_val = evaluation.rebuild_sql_val(p_sql)
        p_sql_col = evaluation.rebuild_sql_col(p_valid_col_units, p_sql_val, kmap)
        return p_sql_col

    @classmethod
    def strip_aliases(cls, query_toks):
        final_sql = []
        invalid = False
        for query_tok in query_toks:
            if query_tok != '.' and '.' in query_tok:
                # invalid sql; didn't use table alias in join
                final_sql.extend(query_tok.replace('.',' . ').split())
                invalid = True
            else:
                final_sql.append(query_tok)
        if 'from' in final_sql:
            sel = final_sql[:final_sql.index('from')]
            all_aliases = Counter([t for t in final_sql if re.match(r't\d+', t)])
            sel_aliases = Counter([t for t in sel if re.match(r't\d+', t)])
            if '*' in sel and len(all_aliases) > len(sel_aliases):
                m = all_aliases.most_common()[-1][0]
                final_sql[final_sql.index('*')] = '{}.*'.format(m)
        return final_sql, invalid

    @classmethod
    def tokenize_question(cls, orig_question_toks, bert):
        question = '___'.join(orig_question_toks).lower()
        for f, t in bad_question_replace:
            question = question.replace(f, t)
        question = ' '.join(question.split('___'))
        question_toks = bert.tokenize(question)
        return question_toks

    @classmethod
    def build_contexts(cls, query_norm_toks, g_values, db, bert, max_lim=512):
        columns = []
        for table_id, (to, t) in enumerate(zip(db['table_names_original'] + ['NULL'], db['table_names'] + ['NULL'])):
            # insert a NULL table at the end
            columns += [{'oname': '*', 'name': '*', 'type': 'all', 'key': '{}.*'.format(to).replace('NULL.', '').lower(), 'table_name': t.lower()}]
            keys = set(db['primary_keys'])
            for a, b in db['foreign_keys']:
                keys.add(a)
                keys.add(b)
            for i, ((tid, co), (_, c), ct) in enumerate(zip(db['column_names_original'], db['column_names'], db['column_types'])):
                ct = ct if i not in keys else 'key'
                if tid == table_id:
                    columns.append({
                        'oname': co, 'name': c, 'type': ct,
                        'key': '{}.{}'.format(to, co).lower(),
                        'table_name': t.lower(),
                    })

        key2col = {col['key']: col for col in columns}

        question_context = [bert.cls_token]
        for t in query_norm_toks:
            if t in key2col:
                col = key2col[t]
                question_context.extend(bert.tokenize('[ {} {} : {} ]'.format(col['type'], col['table_name'], col['name'])))
            else:
                question_context.extend(bert.tokenize(t))
        question_context.append(bert.sep_token)
        for v in g_values:
            question_context.extend(bert.tokenize(' '.join(v)))
            question_context.append(';')
        if question_context[-1] == ';':
            question_context[-1] = bert.sep_token

        if len(question_context) > max_lim:
            raise Exception('question context of {} > {} is too long!'.format(len(question_context), max_lim))
        return question_context, columns

    @classmethod
    def make_column_cands(cls, context):
        return context

    @classmethod
    def get_fuzzy_pointer(cls, token, cands):
        for i, c in enumerate(cands):
            if c == token or c == token + 's' or c + 's' == token:
                return i
        return None

    @classmethod
    def make_sup_question(cls, question_toks, question_cands, bert, voc):
        utt = {}
        utt['column_cands'] = cands = []
        utt['column_pointer'] = pointer = []
        utt['column_toks'] = toks = []

        stemmer = PorterStemmer()
        for t in [t.lower() for t in question_toks]:
            i = cls.get_fuzzy_pointer(stemmer.stem(t), [stemmer.stem(c) for c in question_cands])
            if i is None:
                toks.append(t)
            else:
                toks.append('pointer')
            pointer.append(i)
        toks.append('EOS')
        pointer.append(None)
        voc.word2index(toks, train=True)
        return utt

    @classmethod
    def make_example(cls, ex, bert, utt_voc, conv, train=False):
        db_id = ex['db_id']

        invalid = False
        try:
            # normalize query
            query_norm = conv.convert_tokens(ex['query_toks'], ex['query_toks_no_value'], db_id)
        except Exception as e:
            print('preprocessing error')
            print(ex['query'])
            return None

        if query_norm is None:
            return None
        query_norm_toks = query_norm.split()

        query_recov = g_values = None
        try:
            query_recov = conv.recover(query_norm, db_id)
            em, g_sql, r_sql = conv.match(ex['query'], query_recov, db_id)
            if not em:
                invalid = True
            g_values = cls.align_values(ex['query_toks_no_value'], ex['query_toks'])
        except ValueAlignmentException as e:
            print(ex['query'])
            print(repr(e))
            invalid = True
        except QueryBuildError as e:
            print(ex['query'])
            print(repr(e))
            invalid = True
        except Exception as e:
            print(e)
            invalid = True
            raise

        # make utterance
        question_toks = cls.tokenize_question(ex['question_toks'], bert)
        # print(bert.convert_tokens_to_string(question_toks))

        # encode tables
        try:
            question_context, columns = cls.build_contexts(query_norm_toks, g_values, conv.database_schemas[db_id], bert)
        except Exception as e:
            print(e)
            return None
        # print(bert.convert_tokens_to_string(question_context))

        new = dict(
            id=ex['id'],
            columns=columns,
            db_id=db_id, 
            question=ex['question'],
            g_question_toks=question_toks,
            g_sql=g_sql,
            query=ex['query'],
            g_values=g_values,
            question_context=question_context,
            invalid=invalid,
            cands_question=cls.make_column_cands(question_context),
        )

        if train and not invalid:
            new['sup_question'] = cls.make_sup_question(question_toks, new['cands_question'], bert, utt_voc)
            # print(new['sup_question']['column_toks'])
        return new

    @classmethod
    def recover_slots(cls, pointer, candidates, eos):
        if eos in pointer:
            pointer = pointer[:pointer.index(eos)+1]
        toks = []
        for i, p in enumerate(pointer):
            c = candidates[p]
            toks.append(c)
        if 'EOS' in toks:
            toks = toks[:toks.index('EOS')]
        return toks

    @classmethod
    def recover_question(cls, pointer, candidates, voc):
        toks = cls.recover_slots(pointer, candidates, eos=voc.word2index('EOS'))
        return ' '.join([t.replace(' id', '') for t in toks]).split()

    @classmethod
    def make_cands(cls, ex, utt_voc):
        return utt_voc._index2word + ex['cands_question']

    @classmethod
    def make_question_pointer(cls, sup_question, utt_cands, utt_voc):
        # map utterance
        utt_pointer = []
        for w, p in zip(sup_question['column_toks'], sup_question['column_pointer']):
            if p is None:
                # this is a vocab word
                utt_pointer.append(utt_voc.word2index(w))
            else:
                # this is a column, need to add offset for vocab candidates
                utt_pointer.append(p + len(utt_voc))
        for i in utt_pointer:
            assert i < len(utt_cands)

        toks = cls.recover_question(utt_pointer, utt_cands, voc=utt_voc)
        return utt_pointer

    @classmethod
    def from_file(cls, root, dcache, debug=False):
        conv = converter.Converter(os.path.join(root, 'tables.json'))

        splits = {}
        for k in ['train', 'dev']:
            with open(os.path.join(root, '{}.json'.format(k)), 'rb') as f:
                splits[k] = []
                for ex in json.load(f):
                    ex['query_orig'] = ex['query']
                    splits[k].append(ex)
                    if debug and len(splits[k]) > 100:
                        break
    
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL, cache_dir=dcache)

        utt_voc = Vocab(['PAD', 'EOS', 'GO'])

        # make contexts and populate vocab
        for s, data in splits.items():
            proc = []
            for i, ex in enumerate(tqdm.tqdm(data, desc='preprocess {}'.format(s))):
                ex['id'] = '{}/{}'.format(ex['db_id'], i)
                new = cls.make_example(ex, tokenizer, utt_voc, conv, train=s=='train')
                if new is not None and (s != 'train' or not new['invalid']):
                    proc.append(new)
            splits[s] = proc
    
        # make candidate list using vocab
        for s, data in splits.items():
            for ex in data:
                ex['cands_question'] = cls.make_cands(ex, utt_voc)
            splits[s] = data
    
        # make pointers for training data
        for ex in splits['train']:
            ex['pointer_question'] = cls.make_question_pointer(ex['sup_question'], ex['cands_question'], utt_voc)
    
        # look up pretrained word embeddings
        emb = E.ConcatEmbedding([E.GloveEmbedding(), E.KazumaCharEmbedding()], default='zero')
        utt_emb = torch.tensor([emb.emb(w) for w in utt_voc._index2word])
        ext = dict(utt_voc=utt_voc, utt_emb=utt_emb)
        return splits, ext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data', default='spider')
    args = parser.parse_args()

    proc = SQLDataset.from_file(os.path.join('data', args.data), 'cache', debug=args.debug)
    torch.save(proc, 'cache/data_sql2nl_spider.debug.pt' if args.debug else 'cache/data_sql2nl_spider.pt')
