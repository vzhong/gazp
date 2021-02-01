import os
import re
import numpy as np
import editsql_postprocess
from model.ans2sql import Module as Base, SQLDataset
from collections import defaultdict, Counter


class Module(Base):

    @classmethod
    def reduce_cols(cls, toks):
        # renumber cols and strip out types
        cols = set([t for t in toks if '_col_' in t and 'key' not in t and '*' not in t])
        mapping = {o: 'gen_col_{}'.format(i) for i, o in enumerate(sorted(list(cols)))}
        keep = [mapping.get(t, t) for t in toks]
        return keep
    
    @classmethod
    def reduce_aggs(cls, toks):
        keep = []
        for tt in toks:
            if tt in {'avg', 'min', 'max', 'count', 'sum'}:
                tt = 'agg'
            keep.append(tt)
        return keep
    
    @classmethod
    def reduce_order(cls, toks):
        keep = []
        for tt in toks:
            if tt in {'asc', 'desc'}:
                tt = 'order'
            keep.append(tt)
        return keep
    
    @classmethod
    def reduce_op(cls, toks):
        keep = []
        for tt in toks:
            if tt in {'=', '<', '>', 'like', '<=', '>=', '<>', '!='}:
                tt = 'op'
            keep.append(tt)
        return keep
    
    @classmethod
    def templatize(cls, col_map, query_norm_toks):
        parent = cls.__mro__[1].templatize(col_map, query_norm_toks)
        orig = parent.split()
        r_cols = cls.reduce_cols(orig)
        r_aggs = cls.reduce_aggs(r_cols)
        r_order = cls.reduce_order(r_aggs)
        r_op = cls.reduce_op(r_order)
        template = ' '.join(r_op)
        template = template.replace(' op op ', ' op ')
        return template

    @classmethod
    def detemplatize(cls, col_map, template):
        # map ops
        last_col = None
        temp_ops = []
        for i, w in enumerate(template):
            if w in col_map:
                last_col = col_map[w]
            if w == 'op':
                assert last_col
                t = last_col['type']
                if t in {'key'}:
                    supported = ['=']
                    probs = [1]
                elif t in {'text', 'others'}:
                    supported = ['=', '< >', 'like']
                    probs = [0.7, 0.2, 0.1]
                elif t in {'boolean'}:
                    supported = ['=', '< >']
                    probs = [0.6, 0.4]
                elif t in {'number', '*', 'time'}:
                    # * is count (*)
                    supported = ['=', '>', '> =', '<', '< =', '< >']
                    probs = [0.4, 0.15, 0.1, 0.15, 0.1, 0.1]
                    if 'phone' in last_col['name'].lower():
                        supported = ['=', '< >']
                        probs = [0.7, 0.3]
                else:
                    raise Exception('Unsupported type {} for template {}'.format(t, template))
                assert np.isclose(sum(probs), 1), 'invalid probs sum {} == {}'.format(probs, sum(probs))
                op = np.random.choice(supported, p=probs)
                temp_ops.extend(op.split())
            else:
                temp_ops.append(w)

        # map order
        temp_order = []
        for t in temp_ops:
            if t == 'order':
                supported = ['asc', 'desc']
                probs = [0.5, 0.5]
                order = np.random.choice(supported, p=probs)
                temp_order.append(order)
            else:
                temp_order.append(t)

        # map aggs
        next_col = None
        temp_agg = []
        for w in reversed(temp_order):
            if w in col_map:
                next_col = col_map[w]
            if w == 'agg':
                assert next_col
                t = next_col['type']
                if t in {'key', 'text', '*', 'boolean', 'others'}:
                    supported = ['count']
                    probs = [1]
                elif t in {'number', 'time'}:
                    # * is count (*)
                    supported = ['avg', 'min', 'max', 'count', 'sum']
                    probs = [0.25, 0.25, 0.25, 0.05, 0.2]
                else:
                    raise Exception('Unsupported operator {} in template{}'.format(t, template))
                assert sum(probs) == 1
                agg = np.random.choice(supported, p=probs)
                temp_agg.append(agg)
            else:
                temp_agg.append(w)
        out = temp_agg = list(reversed(temp_agg))

        if 'op' in out or 'agg' in out or 'order' in out:
            print('Could not complete detemplatize for {}'.format(out))
            import pdb; pdb.set_trace()
        return out

    @classmethod
    def get_coarse_type(cls, t):
        return t if t in {'key', '*'} else 'gen'

    @classmethod
    def process_supports(cls, proc_cols):
        supports = defaultdict(set)
        for db_id, cols in proc_cols.items():
            num_types = Counter()
            for c in cols:
                num_types[cls.get_coarse_type(c['type'])] += 1
            for k, v in num_types.items():
                for i in range(v):
                    supports[db_id].add('{}_col_{}'.format(k, i))
        return supports
    
    @classmethod
    def sample_one(cls, db_root, stats, db_ids, schema_tokens, column_names, database_schemas, proc_cols, supports):
        # sample a template
        templates = sorted(list(stats.keys()))
        template_probs = [stats[k] for k in templates]
    
        template = np.random.choice(templates, p=template_probs)
    
        ex = dict(template=template, p_template=stats[template])
    
        # gather databases that can support this template
        vcols = {t for t in template.split() if '_col_' in t}
        supported_dbs = []
        for db_id, supported_cols in supports.items():
            if not vcols - supported_cols:
                # every col is supported
                supported_dbs.append(db_id)
        supported_dbs.sort()
        if not supported_dbs:
            raise Exception('Could not find valid database for {}'.format(vcols))
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
        col_map['*'] = dict(name='*', table_name= '', key='*', type='*')
        mapping = {}
        for c in cols_shuffled:
            t = cls.get_coarse_type(c['type'])
            idx = sum(x.startswith(t) for x in mapping.keys())
            mapping['{}_col_{}'.format(t, idx)] = c['key']
    
        # map columns
        coarse_column_mapped_query = [mapping.get(t, t).lower() for t in template.split()]

        column_mapped_query = cls.detemplatize(col_map, coarse_column_mapped_query)
        # print(coarse_column_mapped_query)
        # print(column_mapped_query)
        # import pdb; pdb.set_trace()

        # map values
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
