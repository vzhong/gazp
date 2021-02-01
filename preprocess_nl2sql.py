import os
import re
import json
import tqdm
import torch
import sqlite3
import converter
import argparse
import embeddings as E
from vocab import Vocab
from collections import defaultdict, Counter
from transformers import DistilBertTokenizer
from eval_scripts import evaluation

import editsql_preprocess
import editsql_postprocess


BERT_MODEL = 'cache/bert'
SQL_PRIMITIVES = {'select', 'from', 'not', 'in', 'where', 'max', 'min', 'avg'}


bad_query_replace = [
    ('ryan___goodwin', 'rylan___goodwin'),
    ('distric_', 'district_'),
    ('districtt_', 'district_'),
    ('northcarolina', 'north___carolina'),
    ('beetle___!', 'beetle'),
    ('caribbean', 'carribean'),
    ('noth', 'north'),
    ('asstprof', 'assistant___professor'),
    ('parallax', 'puzzling'),
    ('region0', 'bay___area'),
    ('timothy', 'timmothy'),
    ('engineering', 'engineer'),
    ('goergia', 'georgia'),
    ('director_name0', 'kevin___spacey'),
    ('actor_name0', 'kevin___spacey'),
    ('category_category_name0', 'mgm___grand___buffet'),
]

bad_question_replace = bad_query_replace
bad_question_replace += [
    ('_one_', '_1_'),
    ('_two_', '_2_'),
    ('_three_', '_3_'),
    ('_four_', '_4_'),
    ('_five_', '_5_'),
    ('_internation_', '_international_'),
]


value_replace = {'_'+k+'_': '_'+v+'_' for k, v in {
    'usa': 'us',
    'africa': 'african',
    'europe': 'european',
    'asia': 'asian',
    'france': 'french',
    'italy': 'italian',
    '2014': '2013',
    'cat': 'cats',
    'dog': 'dogs',
    'male': 'males',
    'female': 'females',
    'student': 'students',
    'engineer': 'engineers',
    'states': 'us',
    'united': 'us',
    'y': 'yes',
    'n': 'no',
    'herbs': 'herb',
    'canada': 'canadian',
    'la': 'louisiana',
    '##ie': '##ies',
    'fl': 'florida',
    'australia': 'australian',
    'professor': 'professors',
    'drive': 'drives',
    'usa': 'united',
    'instructor': 'instructors',
    'completed': 'complete',
    'nominated': 'nomination',
    'game': 'games',
    'card': 'cards',
    'park': 'parking',
    'room': 'rooms',
}.items()}


class ValueAlignmentException(Exception):
    pass


class QueryBuildError(Exception):
    pass


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
                        # if '_' in yk and 'mk_man' not in yk and 'pu_man' not in yk or 't1' in yk or 't2' in yk or 't3' in yk or 't4' in yk:
                        #     break
                        myafter = None if k+1 == len(yes_value) else yes_value[k+1].lower()
                        if myafter == after:
                            candidates.append((j, k+1))
                            break
            if len(candidates) == 0:
                raise ValueAlignmentException('Cannot align values: {}'.format(yes_value))
            candidates.sort(key=lambda x: x[1] - x[0])
            return candidates[0]

        values = []
        num_slots = 0
        for i, t in enumerate(no_value):
            t = t.lower()
            if t in {'value', 'limit_value'}:
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
    def build_contexts(cls, question_toks, db, bert):
        query_context = []

        for table_id, (to, t) in enumerate(zip(db['table_names_original'] + ['NULL'], db['table_names'] + ['NULL'] + [{}])):
            keys = set(db['primary_keys'])
            for a, b in db['foreign_keys']:
                keys.add(a)
                keys.add(b)

            # insert a NULL table at the end
            columns = [{'oname': '*', 'name': '*', 'type': 'all', 'key': '{}.*'.format(to).replace('NULL.', '').lower()}]
            for i, ((tid, co), (_, c), ct) in enumerate(zip(db['column_names_original'], db['column_names'], db['column_types'])):
                ct = ct if i not in keys else 'key'
                if tid == table_id:
                    columns.append({
                        'oname': co, 'name': c, 'type': ct,
                        'key': '{}.{}'.format(to, co).lower(),
                    })
            query_cols = [c.copy() for c in columns]

            # context for generating queries
            query_context_toks = [bert.cls_token] + question_toks + [bert.sep_token] + bert.tokenize(t) + [bert.sep_token]
            for col in query_cols:
                col['start'] = len(query_context_toks)
                query_context_toks.extend(bert.tokenize('{} : {}'.format(col['type'], col['name'])) + [bert.sep_token])
                col['end'] = len(query_context_toks)
                col['table_id'] = table_id

            query_context.append({
                'oname': to,
                'name': t,
                'columns': query_cols,
                'toks': query_context_toks[:512],
            })
        return query_context

    @classmethod
    def make_column_cands(cls, context):
        cands = []
        for tab in context:
            for col in tab['columns']:
                cands.append(col)
        return cands

    @classmethod
    def make_sup_query(cls, norm_query_toks, cands, values, voc, bert, train=True):
        query = {}
        query['column_pointer'] = pointer = []
        query['column_toks'] = toks = []

        query['value_toks'] = []
        for v in values:
            for t in v:
                query['value_toks'].extend(bert.tokenize(t))
            query['value_toks'].append('SEP')
        query['value_toks'].append('EOS')

        for t in [t.lower() for t in norm_query_toks]:
            matched = False
            if t not in SQL_PRIMITIVES:
                for i, c in enumerate(cands):
                    if t == c['key'] and t:
                        toks.append('pointer')
                        pointer.append(i)
                        matched = True
                        break
            if not matched and (train or t in voc._word2index):
                toks.append(t)
                pointer.append(None)
        toks.append('EOS')
        pointer.append(None)
        voc.word2index(toks, train=True)
        return query

    @classmethod
    def make_example(cls, ex, bert, sql_voc, kmaps, conv, train=False, execute=True, evaluation=False):
        db_id = ex['db_id']
        db_path = os.path.join('data', 'database', db_id, db_id + ".sqlite")

        invalid = False
        if evaluation:
            query_norm = query_norm_toks = em = g_sql = g_query = query_recov = g_values = None
        else:
            try:
                # normalize query
                query_norm = conv.convert_tokens(ex['query_toks'], ex['query_toks_no_value'], db_id)
            except Exception as e:
                print('preprocessing error')
                print(ex['query'])
                return None

            if query_norm is None:
                return None

            query_recov = query_norm_toks = g_values = None
            try:
                query_recov = conv.recover(query_norm, db_id)
                query_norm_toks = query_norm.split()
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

            g_sql = conv.build_sql(ex['query'], db_id)
            g_query = ex['query']

        # make utterance
        question_toks = cls.tokenize_question(ex['question_toks'], bert)
        # print(bert.convert_tokens_to_string(question_toks))

        # encode tables
        query_context = cls.build_contexts(question_toks, conv.database_schemas[db_id], bert)
        # print(bert.convert_tokens_to_string(query_context[0]['toks']))

        new = dict(
            id=ex['id'],
            question=ex['question'],
            db_id=db_id, 
            g_question_toks=question_toks,
            g_sql=g_sql,
            query=g_query,
            g_query_norm=query_norm,
            g_query_recov=query_recov,
            g_values=g_values,
            value_context=[bert.cls_token] + question_toks + [bert.sep_token],
            query_context=query_context,
            invalid=invalid,
            cands_query=cls.make_column_cands(query_context),
        )

        if train and not invalid:
            new['sup_query'] = cls.make_sup_query(query_norm_toks, new['cands_query'], g_values, sql_voc, bert)
            # print(new['sup_query']['column_toks'])
        return new

    @classmethod
    def recover_slots(cls, pointer, candidates, eos, key='key'):
        if eos in pointer:
            pointer = pointer[:pointer.index(eos)+1]
        toks = []
        for i, p in enumerate(pointer):
            c = candidates[p]
            if isinstance(c, dict):
                c = c[key]
            toks.append(c)
        if 'EOS' in toks:
            toks = toks[:toks.index('EOS')]
        return toks

    @classmethod
    def recover_query(cls, pointer, candidates, value_pointer, value_candidates, voc):
        toks = cls.recover_slots(pointer, candidates, key='key', eos=voc.word2index('EOS'))
        value = [value_candidates[p] for p in value_pointer]
        if 'EOS' in value:
            value = value[:value.index('EOS')]
        return toks, value

    @classmethod
    def make_cands(cls, ex, sql_voc):
        query_cands = sql_voc._index2word + ex['cands_query']
        value_cands = sql_voc._index2word + ex['value_context']
        return query_cands, value_cands

    @classmethod
    def make_query_pointer(cls, sup_query, query_cands, value_cands, sql_voc):
        # map slots
        pointer = []

        for w, p in zip(sup_query['column_toks'], sup_query['column_pointer']):
            if p is None:
                # this is a vocab word
                pointer.append(sql_voc.word2index(w))
            else:
                # this is a column, need to add offset for vocab candidates
                pointer.append(p + len(sql_voc))

        for i in pointer:
            assert i < len(query_cands)

        # map values
        value_pointer = []
        for w in sup_query['value_toks']:
            if w not in value_cands:
                if w in value_replace:
                    w = value_replace[w]
                else:
                    w = w + 's'
                if w not in value_cands:
                    # print('OOV word in value {}:\n{}\n{}'.format(w, ex['utterance']['toks'], ex['query_toks']))
                    continue
            value_pointer.append(value_cands.index(w))
        # print(cls.recover_query(pointer, cands, value_pointer, value_cands, voc=sql_voc))
        return pointer, value_pointer

    @classmethod
    def from_file(cls, root, dcache, debug=False):
        conv = converter.Converter()
        kmaps = evaluation.build_foreign_key_map_from_json(os.path.join(root, 'tables.json'))

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

        sql_voc = Vocab(['PAD', 'EOS', 'GO', 'SEP', '`', "'", '1', '%', 'yes', '2', '.', '5', 'f', 'm', 'name', 'song', 't', 'l'])

        # make contexts and populate vocab
        for s, data in splits.items():
            proc = []
            for i, ex in enumerate(tqdm.tqdm(data, desc='preprocess {}'.format(s))):
                ex['id'] = '{}/{}'.format(ex['db_id'], i)
                new = cls.make_example(ex, tokenizer, sql_voc, kmaps, conv, train=s=='train')
                if new is not None and (s != 'train' or not new['invalid']):
                    proc.append(new)
            splits[s] = proc
    
        # make candidate list using vocab
        for s, data in splits.items():
            for ex in data:
                ex['cands_query'], ex['cands_value'] = cls.make_cands(ex, sql_voc)
            splits[s] = data
    
        # make pointers for training data
        for ex in splits['train']:
            ex['pointer_query'], ex['pointer_value'] = cls.make_query_pointer(ex['sup_query'], ex['cands_query'], ex['cands_value'], sql_voc)
    
        # look up pretrained word embeddings
        emb = E.ConcatEmbedding([E.GloveEmbedding(), E.KazumaCharEmbedding()], default='zero')
        sql_emb = torch.tensor([emb.emb(w) for w in sql_voc._index2word])
        ext = dict(sql_voc=sql_voc, sql_emb=sql_emb)
        return splits, ext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data', default='spider')
    args = parser.parse_args()

    proc = SQLDataset.from_file(os.path.join('data', args.data), 'cache', debug=args.debug)
    torch.save(proc, 'cache/data_nl2sql_spider.debug.pt' if args.debug else 'cache/data_nl2sql_spider.pt')
