import os
import re
import json
import tqdm
import torch
import pprint
import utils
import pprint
import sqlite3
import dataset
import argparse
import numpy as np
import editsql_preprocess
import editsql_postprocess
from timed_execute import timed_execute
from preprocess_nl2sql import SQLDataset, ValueAlignmentException
from collections import Counter, defaultdict
from eval_scripts import evaluation


mydir = os.path.dirname(__file__)


class EmptyColumnException(Exception):
    pass


class NoSupportedDBException(Exception):
    pass


class UnknownOpException(Exception):
    pass


class Module:

    @classmethod
    def templatize(cls, col_map, query_norm_toks):
        seen_col = {}
        toks = []
        for t in query_norm_toks:
            if t in col_map:
                if t not in seen_col:
                    idx = sum(x.startswith(col_map[t]['type']) for x in seen_col.values())
                    seen_col[t] = '{}_col_{}'.format(col_map[t]['type'], idx)
                toks.append(seen_col[t])
            else:
                toks.append(t)
        template = ' '.join(toks)
        return template

    @classmethod
    def load_statistics(cls, fname, column_names, schema_tokens, database_schemas, proc_cols):
        with open(fname) as f:
            data = json.load(f)
    
        templates = Counter()
        for ex in tqdm.tqdm(data, desc='stats'):
            db_id = ex['db_id']
            query_no_alias_toks, invalid = SQLDataset.strip_aliases(ex['query_toks_no_value'])
            query_no_alias = ' '.join(query_no_alias_toks)
            try:
                query_norm = editsql_preprocess.parse_sql(query_no_alias, db_id, column_names[db_id], editsql_preprocess.output_vocab_without_from, schema_tokens[db_id], database_schemas[db_id])
            except:
                continue
            query_norm_toks = query_norm.split()
            col_map = {c['key']: c for c in proc_cols[db_id]}
            template = cls.templatize(col_map, query_norm_toks)
            templates[template] += 1
        total = sum(templates.values())
        dist = Counter({k: v/total for k, v in templates.items()})
        return dist
    
    @classmethod
    def process_cols(cls, db_ids, database_schemas):
        all_cols = {}
        for db_id in db_ids:
            db = database_schemas[db_id]
            schema = database_schemas[db_id]
            all_cols[db_id] = cols = []
            keys = set(schema['primary_keys'])
            for a, b in schema['foreign_keys']:
                keys.add(a)
                keys.add(b)
            for i, ((table_idx, name), typ) in enumerate(zip(schema['column_names_original'], schema['column_types'])):
                if table_idx != -1:
                    table_name = schema['table_names_original'][table_idx]
                    cols.append(dict(
                        name=name,
                        type=typ if i not in keys else 'key',
                        table_name=table_name,
                        key='{}.{}'.format(table_name, name).lower(),
                    ))
            for name in schema['table_names_original']:
                cols.append(dict(name='*', table_name=name, type='*', key='{}.*'.format(name).lower()))
        return all_cols
    
    @classmethod
    def process_supports(cls, proc_cols):
        supports = defaultdict(set)
        for db_id, cols in proc_cols.items():
            num_types = Counter([c['type'] for c in cols])
            for k, v in num_types.items():
                for i in range(v):
                    supports[db_id].add('{}_col_{}'.format(k, i))
        return supports
    
    @classmethod
    def sample_value(cls, col, op, db_path):
        if op == 'count':
            v = np.random.randint(1, 10)
            return str(v), [str(v)]

        if col is None:
            raise EmptyColumnException('cannot sample value for empty column')
    
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        q = 'select {} from {}'.format(col['name'], col['table_name'])
        try:
            cursor.execute(q)
            res = [str(r[0]).split('.')[0] for r in cursor.fetchall()]
            res = [r for r in res if r]
        except Exception as e:
            raise EmptyColumnException('cannot read column content: {}'.format(e))
    
        if not res:
            raise EmptyColumnException('column {} is empty!'.format(col['key']))
    
        val_toks = []
        if op is None:
            if col['type'] in {'id', 'number'}:
                val = np.random.choice(res)
                val_toks.append(val)
            else:
                v = np.random.choice(res)
                val = "'{}'".format(v)
                val_toks.append("'{}".format(v))
                val_toks.append("'")
        elif op in {'max', 'min', 'avg', 'sum'}:
            val = np.random.choice(res)
            val_toks.append(val)
        else:
            raise UnknownOpException('Unknown op {}'.format(op))
        return str(val), [str(v) for v in val_toks]
    
    
    @classmethod
    def sample_one(cls, db_root, stats, allowed_db_ids, schema_tokens, column_names, database_schemas, proc_cols, supports):
        # sample a template
        templates = sorted(list(stats.keys()))
        template_probs = [stats[k] for k in templates]
    
        template = np.random.choice(templates, p=template_probs)
    
        ex = dict(template=template, p_template=stats[template])
    
        # gather databases that can support this template
        vcols = {t for t in template.split() if '_col_' in t}
        supported_dbs = []
        for db_id, supported_cols in supports.items():
            if db_id in allowed_db_ids and not vcols - supported_cols:
                # every col is supported
                supported_dbs.append(db_id)
        if not supported_dbs:
            raise NoSupportedDBException(template)
        supported_dbs.sort()
        ex['num_supported_db'] = len(supported_dbs)
        ex['p_supported_db'] = 1/len(supported_dbs)
    
        # sample a database at random
        db_id = np.random.choice(supported_dbs)
        db_path = os.path.join(db_root, db_id, db_id + '.sqlite')
        cols = proc_cols[db_id]
    
        # make a random mapping of columns
        cols_shuffled = cols[:]
        np.random.shuffle(cols_shuffled)
        col_map = {c['key']: c for c in proc_cols[db_id]}
        col_map['*'] = '*'
        mapping = {}
        for c in cols_shuffled:
            idx = sum(x.startswith(c['type']) for x in mapping.keys())
            mapping['{}_col_{}'.format(c['type'], idx)] = c['key']
    
        # insert mapping map into query
        column_mapped_query = [mapping.get(t, t).lower() for t in template.split()]
        value_mapped_query = []
        last_col = None
        last_op = None
        query = ''
        for i, t in enumerate(column_mapped_query):
            if t in col_map:
                last_col = col_map[t]
                if i-2 >= 0 and column_mapped_query[i-1] == '(':
                    last_op = column_mapped_query[i-2]
                else:
                    last_op = None
            if t == 'value':
                val, val_toks = cls.sample_value(last_col, last_op, db_path)
                query += ' ' + val
                value_mapped_query.extend(val_toks)
            elif t == 'limit_value':
                query += ' limit 1'
                value_mapped_query.extend(['limit', '1'])
            else:
                query += ' ' + t
                value_mapped_query.append(t)
    
        ex['recov'] = editsql_postprocess.postprocess_one(query, database_schemas[db_id])
        ex['recov'] = re.sub('\s([0-9a-zA-Z_]+\.)\*\s', '', ex['recov'])
        ex['values'] = SQLDataset.align_values(column_mapped_query, value_mapped_query)
    
        ex['column_mapped'] = column_mapped_query
        ex['value_mapped'] = value_mapped_query
        ex['norm_query'] = query.strip().split()
        ex['db_id'] = db_id
        return ex
    
    @classmethod
    def execute_query(cls, db_root, query, database_schemas, silent=False):
        db_id = query['db_id']
        db_path = os.path.join(db_root, db_id, db_id + '.sqlite')
        query_recov = query['recov']
        if 't5' in query_recov:
            return None
        query['query_toks'] = query_recov.replace('.', ' . ').split()
        query['query_toks_no_value'] = editsql_postprocess.postprocess_one(' '.join(query['column_mapped']), database_schemas[db_id]).replace('limit 1', 'limit_value').replace(' 1', ' value').replace('.', ' . ').split(' ')
        schema = evaluation.Schema(evaluation.get_schema(db_path))
        g_raw_res = timed_execute(db_path, query_recov, timeout=1, sleep=0.001, silent=silent)
        return g_raw_res
    
    @classmethod
    def sample_executable_queries(cls, db_root, num, db_ids, schema_tokens, column_names, database_schemas, proc_cols, supports, stats, args=None, silent=False):
        queries = []
        pbar = tqdm.tqdm(desc='generate', total=num)
        while len(queries) < num:
            try:
                query = cls.sample_one(db_root, stats, db_ids, schema_tokens, column_names, database_schemas, proc_cols, supports)
            except NoSupportedDBException as e:
                pass
            except EmptyColumnException as e:
                pass
            except ValueAlignmentException as e:
                pass
            try:
                res = cls.execute_query(db_root, query, database_schemas, silent=silent)
            except Exception as e:
                print('failed to execute query: {}'.format(' '.join(query['value_mapped'])))
                print(e)
                print()
                continue
            if res:
                queries.append(query)
                pbar.update(1)
        return queries
