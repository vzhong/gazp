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
from preprocess_nl2sql import SQLDataset as Base, ValueAlignmentException, QueryBuildError, value_replace

import editsql_preprocess
import editsql_postprocess


BERT_MODEL = 'cache/bert'


class SQLDataset(Base):

    @classmethod
    def build_contexts(cls, question_toks, prev_query_toks, db, bert):
        query_context = []
        nl_map = {}
        for table_id, (to, t) in enumerate(zip(db['table_names_original'] + ['NULL'], db['table_names'] + ['NULL'] + [{}])):
            for i, ((tid, co), (_, c), ct) in enumerate(zip(db['column_names_original'], db['column_names'], db['column_types'])):
                nl_map['{}.{}'.format(to, co).lower()] = '{} {}'.format(t, c)
        prev_query = ' '.join([nl_map.get(t, t) for t in prev_query_toks]).replace('_', ' ')
        prev_query_toks = bert.tokenize(prev_query)

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
            query_context_toks = [bert.cls_token] + prev_query_toks + [bert.sep_token] + question_toks +  [bert.sep_token] + bert.tokenize(t) + [bert.sep_token]
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
    def tokenize_query(cls, query):
        toks = []
        curr = []
        delims = '()!='
        aug = query.rstrip(';').replace('"', "'")
        for c in delims:
            aug = aug.replace(c, ' {} '.format(c))
        toks = []
        for t in aug.split():
            if t.endswith("'"):
                toks.append(t.rstrip("'"))
                toks.append("'")
            elif '.' in t and t.lower().startswith('t') and len(t.split('.')) == 2:
                start, end = t.split('.')
                toks.append(start)
                toks.append('.')
                toks.append(end)
            else:
                toks.append(t)
        toks_no_value = []
        in_str = False
        for t in toks:
            if t.startswith("'") and t != "'":
                in_str = True
                continue
            if in_str and t == "'":
                in_str = False
                toks_no_value.append('value')
                continue
            if not in_str:
                try:
                    float(t)
                except:
                    toks_no_value.append(t.lower())
                else:
                    toks_no_value.append('value')
        return toks, toks_no_value

    @classmethod
    def make_example(cls, ex, bert, sql_voc, kmaps, conv, train=False, execute=True):
        db_id = ex['db_id']
        db_path = os.path.join('data', 'database', db_id, db_id + ".sqlite")

        ex['query_toks'], ex['query_toks_no_value'] = cls.tokenize_query(ex['query'])

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

        if ex['prev'] is not None:
            prev_query_toks, prev_query_toks_no_value = cls.tokenize_query(ex['prev']['query'])
            prev_query_norm = conv.convert_tokens(prev_query_toks, prev_query_toks_no_value, db_id)
            if prev_query_norm is None:
                prev_query_norm = 'none'
        else:
            prev_query_norm = 'none'

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
            raise
        except QueryBuildError as e:
            print(ex['query'])
            print(repr(e))
            invalid = True
            raise
        except Exception as e:
            print(e)
            invalid = True
            raise

        # make utterance
        question_toks = cls.tokenize_question(ex['utterance'].split(), bert)
        # print(bert.convert_tokens_to_string(question_toks))

        # encode tables
        query_context = cls.build_contexts(question_toks, prev_query_norm.split(), conv.database_schemas[db_id], bert)
        # print(bert.convert_tokens_to_string(query_context[0]['toks']))

        g_sql = conv.build_sql(ex['query'], db_id)
        new = dict(
            id=ex['id'],
            db_id=db_id, 
            question=ex['utterance'],
            g_question_toks=question_toks,
            g_sql=g_sql,
            query=ex['query'],
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
    def from_file(cls, root, dspider, dcache, debug=False):
        train_database, dev_database = editsql_preprocess.read_db_split(dspider)
        conv = converter.Converter()
        kmaps = evaluation.build_foreign_key_map_from_json(os.path.join(dspider, 'tables.json'))

        splits = {}
        for k in ['train', 'dev']:
            with open(os.path.join(root, '{}.json'.format(k)), 'rb') as f:
                splits[k] = []
                for ex in json.load(f):
                    splits[k].append(ex)
                    if debug and len(splits[k]) > 100:
                        break
    
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL, cache_dir=dcache)

        sql_voc = Vocab(['PAD', 'EOS', 'GO', 'SEP', '`', "'", '1', '%', 'yes', '2', '.', '5', 'f', 'm', 'name', 'song', 't', 'l'])

        # make contexts and populate vocab
        for s, data in splits.items():
            proc = []
            for i, ex in enumerate(tqdm.tqdm(data, desc='preprocess {}'.format(s))):
                for turn_i, turn in enumerate(ex['interaction']):
                    turn['id'] = '{}/{}:{}'.format(ex['database_id'], i, turn_i)
                    turn['db_id'] = ex['database_id']
                    turn['prev'] = ex['interaction'][turn_i-1] if turn_i > 0 else None
                    new = cls.make_example(turn, tokenizer, sql_voc, kmaps, conv, train=s=='train')
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
    parser.add_argument('--data', default='sparc')
    args = parser.parse_args()

    proc = SQLDataset.from_file(os.path.join('data', args.data), os.path.join('data', 'spider'), 'cache', debug=args.debug)
    torch.save(proc, 'cache/data_nl2sql_sparc_sparc.debug.pt' if args.debug else 'cache/data_nl2sql_sparc_sparc.pt')
