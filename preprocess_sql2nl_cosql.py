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
import preprocess_nl2sql_cosql as preprocess_nl2sql
from vocab import Vocab
from collections import defaultdict, Counter
from transformers import DistilBertTokenizer
from eval_scripts import evaluation
from preprocess_sql2nl import SQLDataset as Base, BERT_MODEL
from nltk.stem.porter import PorterStemmer

import editsql_preprocess
import editsql_postprocess


class SQLDataset(Base):

    @classmethod
    def build_contexts(cls, query_norm_toks, prev_query_toks, g_values, db, bert, max_lim=512):
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

        for t in prev_query_toks:
            if t in key2col:
                col = key2col[t]
                question_context.extend(bert.tokenize('[ {} {} : {} ]'.format(col['type'], col['table_name'], col['name'])))
            else:
                question_context.extend(bert.tokenize(t))
        question_context.append(bert.sep_token)

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
    def make_example(cls, ex, bert, utt_voc, conv, train=False):
        db_id = ex['db_id']

        ex['query_toks'], ex['query_toks_no_value'] = preprocess_nl2sql.SQLDataset.tokenize_query(ex['query'])
        invalid = False
        try:
            # normalize query
            query_norm = conv.convert_tokens(ex['query_toks'], ex['query_toks_no_value'], db_id)
        except Exception as e:
            print('preprocessing error')
            print(ex['query'])
            raise
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
        question_toks = cls.tokenize_question(ex['utterance'].split(), bert)
        # print(bert.convert_tokens_to_string(question_toks))

        if ex['prev'] is not None:
            prev_query_toks, prev_query_toks_no_value = preprocess_nl2sql.SQLDataset.tokenize_query(ex['prev']['query'])
            prev_query_norm = conv.convert_tokens(prev_query_toks, prev_query_toks_no_value, db_id)
            if prev_query_norm is None:
                prev_query_norm = 'none'
        else:
            prev_query_norm = 'none'

        # encode tables
        try:
            question_context, columns = cls.build_contexts(query_norm_toks, prev_query_norm.split(), g_values, conv.database_schemas[db_id], bert)
        except Exception as e:
            print(e)
            return None
        # print(bert.convert_tokens_to_string(question_context))

        new = dict(
            id=ex['id'],
            query_norm=query_norm,
            prev_query_norm=prev_query_norm,
            columns=columns,
            db_id=db_id, 
            question=ex['utterance'],
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
    def from_file(cls, root, dspider, dcache, debug=False):
        train_database, dev_database = editsql_preprocess.read_db_split(dspider)
        conv = converter.Converter(os.path.join(dspider, 'tables.json'))

        splits = {}
        for k in ['train', 'dev']:
            with open(os.path.join(root, '{}.json'.format(k)), 'rb') as f:
                splits[k] = []
                for ex in json.load(f):
                    splits[k].append(ex)
                    if debug and len(splits[k]) > 100:
                        break
    
        tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL, cache_dir=dcache)

        utt_voc = Vocab(['PAD', 'EOS', 'GO'])

        # make contexts and populate vocab
        for s, data in splits.items():
            proc = []
            for i, ex in enumerate(tqdm.tqdm(data, desc='preprocess {}'.format(s))):
                for turn_i, turn in enumerate(ex['interaction']):
                    turn['id'] = '{}/{}:{}'.format(ex['database_id'], i, turn_i)
                    turn['db_id'] = ex['database_id']
                    turn['prev'] = ex['interaction'][turn_i-1] if turn_i > 0 else None
                    new = cls.make_example(turn, tokenizer, utt_voc, conv, train=s=='train')
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
    parser.add_argument('--data', default='cosql')
    args = parser.parse_args()

    proc = SQLDataset.from_file(os.path.join('data', args.data), os.path.join('data', 'spider'), 'cache', debug=args.debug)
    torch.save(proc, 'cache/data_sql2nl_sparc_cosql.debug.pt' if args.debug else 'cache/data_sql2nl_sparc_cosql.pt')
