import os
import re
import json
import tqdm
import torch
import sqlite3
import argparse
import embeddings as E
from vocab import Vocab
from collections import defaultdict, Counter
from transformers import DistilBertTokenizer
from eval_scripts import evaluation

import editsql_preprocess
import editsql_postprocess


BERT_MODEL = 'distilbert-base-uncased'
SQL_PRIMITIVES = {'select', 'from', 'not', 'in', 'where', 'max', 'min', 'avg'}


bad_query_replace = [
    ('ryan___goodwin', 'rylan___goodwin'),
    ('distric', 'district'),
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

value_replace = {
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
    '1000': '800',
    'interational': 'internation',
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
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '10': 'ten',
    'instructor': 'instructors',
    'completed': 'complete',
    'nominated': 'nomination',
    'game': 'games',
    'card': 'cards',
    'park': 'parking',
    'room': 'rooms',
}


class SQLDataset:

    def __init__(self):
        pass


    @classmethod
    def load_db_content(cls, schemas):
        content = {}
        for db_name, val in schemas.items():
            db = os.path.join('data', 'database', db_name, db_name + ".sqlite")
            conn = sqlite3.connect(db)
            cursor = conn.cursor()

            content[db_name] = db_content = []
            for i, table in enumerate(val['table_names_original']):
                cols = [c for j, c in val['column_names_original'] if c != '*' and j == i]
                cursor.execute('select * from {} limit 5'.format(table))
                try:
                    res = cursor.fetchall()
                except Exception as e:
                    print(e)
                    res = []
                d = defaultdict(list)
                for row in res:
                    assert len(row) == len(cols), 'Cannot fit\n{}\ninto\n{}'.format(row, cols)
                    for n, c in zip(cols, row):
                        d[n].append(c)
                db_content.append(dict(d))
        return content

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
    def build_contexts(cls, question_toks, db, db_content, bert):
        query_context = []

        for table_id, (to, t, t_content) in enumerate(zip(db['table_names_original'] + ['NULL'], db['table_names'] + ['NULL'], db_content + [{}])):
            # insert a NULL table at the end
            columns = [{'oname': '*', 'name': '*', 'type': 'all', 'key': '{}.*'.format(to).replace('NULL.', '').lower()}]
            for (tid, co), (_, c), ct in zip(db['column_names_original'], db['column_names'], db['column_types']):
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
    def make_sup_query(cls, norm_query_toks, cands, voc, bert):
        query = {}
        query['column_pointer'] = pointer = []
        query['column_toks'] = toks = []

        for t in [t.lower() for t in norm_query_toks]:
            matched = False
            if t not in SQL_PRIMITIVES:
                for i, c in enumerate(cands):
                    if t == c['key'] and t:
                        toks.append('pointer')
                        pointer.append(i)
                        matched = True
                        break
            if not matched:
                toks.append(t)
                pointer.append(None)
        toks.append('EOS')
        pointer.append(None)
        voc.word2index(toks, train=True)
        return query

    @classmethod
    def make_example(cls, ex, bert, sql_voc, column_names, schema_tokens, database_schemas, kmaps, db_contents, train=False, execute=True):
        db_id = ex['db_id']
        db = database_schemas[db_id]
        db_path = os.path.join('data', 'database', db_id, db_id + ".sqlite")
        db_content = db_contents[db_id]

        try:
            # normalize query
            query_no_alias_toks, invalid = cls.strip_aliases(ex['query_toks_no_value'])
            query_no_alias = ' '.join(query_no_alias_toks)
            query_norm = editsql_preprocess.parse_sql(query_no_alias, db_id, column_names[db_id], editsql_preprocess.output_vocab_without_from, schema_tokens[db_id], db)
            query_recov = editsql_postprocess.postprocess_one(query_norm, database_schemas[db_id])
            query_norm_toks = query_norm.split()

            # execute query to get results
            schema = evaluation.Schema(evaluation.get_schema(db_path))
            g_sql = evaluation.get_sql(schema, ex['query'])
            g_sql = cls.build_sql(schema, ex['query'], kmaps[db_id])
            g_res = None
        except Exception as e:
            print(ex['query'])
            print(e)
            return None

        # make utterance
        question_toks = cls.tokenize_question(ex['question_toks'], bert)
        # print(bert.convert_tokens_to_string(question_toks))

        # encode tables
        query_context = cls.build_contexts(question_toks, db, db_content, bert)
        # print(bert.convert_tokens_to_string(query_context[0]['toks']))

        new = dict(
            id=ex['id'],
            db_id=db_id, 
            g_question_toks=question_toks,
            g_sql=g_sql,
            g_query=ex['query'],
            g_query_norm=query_norm,
            g_query_recov=query_recov,
            g_res=g_res,
            query_context=query_context,
            invalid=invalid,
            cands_query=cls.make_column_cands(query_context),
        )

        if train and not invalid:
            new['sup_query'] = cls.make_sup_query(query_norm_toks, new['cands_query'], sql_voc, bert)
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
    def recover_query(cls, pointer, candidates, voc):
        toks = cls.recover_slots(pointer, candidates, key='key', eos=voc.word2index('EOS'))
        return toks

    @classmethod
    def make_cands(cls, ex, sql_voc):
        query_cands = sql_voc._index2word + ex['cands_query']
        return query_cands

    @classmethod
    def make_query_pointer(cls, sup_query, query_cands, sql_voc):
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
        return pointer

    @classmethod
    def from_file(cls, root, dcache, debug=False):
        train_database, dev_database = editsql_preprocess.read_db_split(root)
        schema_tokens = {}
        column_names = {}
        database_schemas = {}
        schema_tokens, column_names, database_schemas = editsql_preprocess.read_database_schema(os.path.join(root, 'tables.json'), schema_tokens, column_names, database_schemas)
        kmaps = evaluation.build_foreign_key_map_from_json(os.path.join(root, 'tables.json'))

        db_content = cls.load_db_content(database_schemas)
    
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
                new = cls.make_example(ex, tokenizer, sql_voc, column_names, schema_tokens, database_schemas, kmaps, db_content, train=s=='train')
                if new is not None and (s != 'train' or not new['invalid']):
                    proc.append(new)
            splits[s] = proc
    
        # make candidate list using vocab
        for s, data in splits.items():
            for ex in data:
                ex['cands_query'] = cls.make_cands(ex, sql_voc)
            splits[s] = data
    
        # make pointers for training data
        for ex in splits['train']:
            ex['pointer_query'] = cls.make_query_pointer(ex['sup_query'], ex['cands_query'], sql_voc)
    
        # look up pretrained word embeddings
        emb = E.ConcatEmbedding([E.GloveEmbedding(), E.KazumaCharEmbedding()], default='zero')
        sql_emb = torch.tensor([emb.emb(w) for w in sql_voc._index2word])
        ext = dict(sql_voc=sql_voc, sql_emb=sql_emb, database_schemas=database_schemas, db_content=db_content, kmaps=kmaps)
        return splits, ext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data', default='spider')
    args = parser.parse_args()

    proc = SQLDataset.from_file(os.path.join('data', args.data), 'cache', debug=args.debug)
    torch.save(proc, 'cache/data_nl2sql_novalue.debug.pt' if args.debug else 'cache/data_nl2sql_novalue.pt')
