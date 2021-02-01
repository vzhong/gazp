import os
import re
import tqdm
import json
import copy
import argparse
import editsql_preprocess
import editsql_postprocess
from eval_scripts import evaluation
from collections import Counter


SQL_PRIMITIVES = editsql_preprocess.output_vocab_with_from


class Converter:

    def __init__(self, tables='data/spider/tables.json', db='data/database'):
        self.tables = tables
        self.db = db
        self.schema_tokens, self.column_names, self.database_schemas = editsql_preprocess.read_database_schema(tables, {}, {}, {})
        self.kmaps = evaluation.build_foreign_key_map_from_json(tables)
        self.evaluator = evaluation.Evaluator()

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
    def strip_values(cls, toks):
        proc = []
        # remove numbers and strings
        for t in toks:
            try:
                float(t)
            except:
                if t.startswith('"') and t.endswith('"'):
                    proc.append('value')
                else:
                    proc.append(t.lower())
            else:
                proc.append('value')
        return proc

    def build_sql(self, query, db_id):
        db_path = os.path.join(self.db, db_id, db_id + ".sqlite")
        schema = evaluation.Schema(evaluation.get_schema(db_path))

        try:
            sql = evaluation.get_sql(schema, query)
        except Exception as e:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            sql = evaluation.EMPTY_QUERY.copy()
            print(e)
        valid_col_units = evaluation.build_valid_col_units(sql['from']['table_units'], schema)
        sql_val = evaluation.rebuild_sql_val(sql)
        sql_col = evaluation.rebuild_sql_col(valid_col_units, sql_val, self.kmaps[db_id])
        return sql_col

    def convert_tokens(self, query_toks, query_toks_no_value, db_id):
        db = self.database_schemas[db_id]
        query_no_alias_toks, invalid = self.strip_aliases(query_toks_no_value)
        query_no_alias = ' '.join(query_no_alias_toks)
        try:
            query_norm = editsql_preprocess.parse_sql(query_no_alias, db_id, self.column_names[db_id], editsql_preprocess.output_vocab_without_from, self.schema_tokens[db_id], db)
        except AssertionError as e:
            print('assertion error in parse_sql', e)
            query_norm = None
        except editsql_preprocess.OOVException as e:
            print('oov', e, 'in', query_no_alias)
            query_norm = None
        except ValueError as e:
            print('valueerror', e, 'in', query_no_alias)
            query_norm = None
        return query_norm

    def recover(self, converted, db_id):
        db = self.database_schemas[db_id]
        query_recov = editsql_postprocess.postprocess_one(converted, db)
        return query_recov

    def match(self, query, query_recov, db_id):
        g_sql = self.build_sql(query, db_id)
        r_sql = self.build_sql(query_recov, db_id)
        em = self.evaluator.eval_exact_match(copy.deepcopy(r_sql), copy.deepcopy(g_sql))
        return em, g_sql, r_sql


def parse_sql(sql_string, db_id, column_names, output_vocab, schema_tokens, schema):
    format_sql = editsql_preprocess.sqlparse.format(sql_string, reindent=True)
    format_sql_2 = editsql_preprocess.normalize_space(format_sql)

    num_from = sum([1 for sub_sql in format_sql_2.split('\n') if sub_sql.startswith('from')])
    num_select = format_sql_2.count('select ') + format_sql_2.count('select\n')

    format_sql_3, used_tables, used_tables_list = editsql_preprocess.remove_from_with_join(format_sql_2)

    format_sql_3 = '\n'.join(format_sql_3)
    format_sql_4 = editsql_preprocess.add_table_name(format_sql_3, used_tables, column_names, schema_tokens)

    format_sql_4 = '\n'.join(format_sql_4)
    format_sql_5 = editsql_preprocess.remove_from_without_join(format_sql_4, column_names, schema_tokens)

    format_sql_5 = '\n'.join(format_sql_5)
    format_sql_final = editsql_preprocess.normalize_final_sql(format_sql_5)

    candidate_tables_id, table_names_original = editsql_preprocess.get_candidate_tables(format_sql_final, schema)

    failure = False
    if len(candidate_tables_id) != len(used_tables):
        failure = True

    print(1)
    print(format_sql)
    print()
    print(2)
    print(format_sql_2)
    print()
    print(3)
    print(format_sql_3)
    print()
    print(4)
    print(format_sql_4)
    print()
    print(5)
    print(format_sql_5)
    print()

    editsql_preprocess.check_oov(format_sql_final, output_vocab, schema_tokens)
    import pdb; pdb.set_trace()
    return format_sql_final


def iterate_spider(data):
    for ex in tqdm.tqdm(data):
        yield ex['query'], ex['db_id']


def iterate_sparc(data):
    new_data = []
    for ex in data:
        for i in ex['interaction']:
            new_data.append(dict(query=i['query'], db_id=ex['database_id']))
    return iterate_spider(new_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsplit', default=os.path.join('data', 'spider', 'dev.json'))
    args = parser.parse_args()

    converter = Converter()
    
    with open(args.fsplit) as f:
        data = json.load(f)

    if 'spider' in args.fsplit:
        iterator = iterate_spider(data)
    elif 'sparc' in args.fsplit:
        iterator = iterate_sparc(data)

    total = matched = 0
    for query, db_id in iterator:
        yes_value, no_value, conv = converter.convert(query, db_id)
        if conv is None:
            em = False
        else:
            recov = converter.recover(conv, db_id)
            em, g_sql, r_sql = converter.match(query.lower(), recov, db_id)
        matched += em
        total += 1
        # if not em:
        #     print('query')
        #     print(query)
        #     print('norm')
        #     print(conv)
        #     print('recov')
        #     print(recov)
        #     import pdb; pdb.set_trace()
    print(matched/total)

