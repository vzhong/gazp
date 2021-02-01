import os
import re
import json
import tqdm
import sqlite3
import numpy as np
import editsql_preprocess
import editsql_postprocess
import preprocess_sql2nl_sparc as preprocess
import preprocess_nl2sql_sparc as preprocess_nl2sql
import preprocess_nl2sql_cosql
from timed_execute import timed_execute
from eval_scripts import evaluation
from collections import Counter, defaultdict
from model.sql2nl import Module as Base, dataset, corpus_bleu, SmoothingFunction, torch
from model.ans2sql import NoSupportedDBException, ValueAlignmentException, EmptyColumnException


class Module(Base):

    def extract_preds(self, out, feat, batch):
        preds = {}
        for utt_dec, ex in zip(out['utt_dec'], batch):
            if self.should_beam_search():
                for beam in utt_dec:
                    beam.toks = preprocess.SQLDataset.recover_question(beam.inds, ex['cands_question'], voc=self.utt_vocab)
                b = self.reranker.rerank(utt_dec, ex)
                inds = b.inds
                toks = b.toks
                raw_scores = None
            else:
                inds = utt_dec.max(1)[1].tolist()
                toks = preprocess.SQLDataset.recover_question(inds, ex['cands_question'], voc=self.utt_vocab)
                raw_scores = utt_dec.to(torch.device('cpu'))
            utt = self.bert_tokenizer.convert_tokens_to_string(toks)
            for c in ex['columns']:
                if ' id' in c['name']:
                    utt = utt.replace(c['name'], c['name'].replace(' id', ''))
            preds[ex['id']] = dict(
                utt_pointer=inds,
                utt_toks=toks,
                utt=utt,
                raw_scores=raw_scores,
            )
        return preds

    def compute_metrics(self, data, preds):
        em = 0
        utt_hyps, utt_refs = [], []
        generated = dataset.Dataset()
        for ex in data:
            p = preds[ex['id']]
            utt_hyps.append(p['utt_toks'])
            utt_refs.append([ex['g_question_toks']])

            # make new example
            db_id = ex['db_id']
            db = self.conv.database_schemas[db_id]
            question_toks = p['utt_toks']
            # make original question
            question = self.bert_tokenizer.convert_tokens_to_string(question_toks)
            # casing
            for val_toks in ex['g_values']:
                val = self.bert_tokenizer.convert_tokens_to_string(val_toks).strip(' \"\'`%')
                question = question.replace(val.lower(), val)

            query_context = preprocess_nl2sql.SQLDataset.build_contexts(question_toks, ex['prev_query_norm'].split(), db, self.bert_tokenizer)

            if 'g_sql' not in ex:
                ex['g_sql'] = self.conv.build_sql(ex['query'], db_id)

            new = dict(
                id=ex['id'],
                db_id=db_id, 
                question=question,
                g_question_toks=question_toks,
                query=ex['query'],
                g_values=ex['g_values'],
                g_sql=ex['g_sql'],
                value_context=[self.bert_tokenizer.cls_token] + question_toks + [self.bert_tokenizer.sep_token],
                query_context=query_context,
                invalid=False,
                cands_query=preprocess_nl2sql.SQLDataset.make_column_cands(query_context),
            )
            new['cands_query'], new['cands_value'] = preprocess_nl2sql.SQLDataset.make_cands(new, self.nl2sql.sql_vocab)
            generated.append(new)

        metrics = {
            'utt_bleu': corpus_bleu(utt_refs, utt_hyps, smoothing_function=SmoothingFunction().method3),
        }
        if not self.training:
            with torch.no_grad():
                self.nl2sql.eval()
                preds = self.nl2sql.run_pred(generated, self.nl2sql.args, verbose=True, desc='cycle_pred', force_non_interactive=True)
            metrics.update(self.nl2sql.compute_official_eval(generated, preds))
        return metrics

    def compute_upperbound(self, dev):
        preds = {}
        for ex in dev:
            utt_toks = preprocess.SQLDataset.recover_question(ex['pointer_question'], ex['cands_question'], voc=self.utt_vocab)
            utt = self.bert_tokenizer.convert_tokens_to_string(utt_toks)
            preds[ex['id']] = dict(
                utt_toks=utt_toks,
                utt=utt,
            )
        self.eval()
        metrics = self.compute_metrics(dev, preds)
        return metrics

    @classmethod
    def load_statistics(cls, fname, SQLSampler, conv, proc_cols):
        with open(fname) as f:
            data = json.load(f)
    
        prevs = Counter()
        curr_given_prevs = defaultdict(lambda: Counter())
        for ex in tqdm.tqdm(data, desc='stats'):
            prev_query = 'none'
            db_id = ex['database_id']
            for turn in ex['interaction']:
                query = turn['query']
                query_toks, query_toks_no_value = preprocess_nl2sql.SQLDataset.tokenize_query(query)
                query_norm = conv.convert_tokens(query_toks, query_toks_no_value, db_id)
                if query_norm is None:
                    continue

                col_map = {c['key']: c for c in proc_cols[db_id]}

                prev_temp = SQLSampler.templatize(col_map, prev_query.split())
                prevs[prev_temp] += 1
                curr_temp = SQLSampler.templatize(col_map, query_norm.split())
                curr_given_prevs[prev_temp][curr_temp] += 1
                prev_query = query_norm
        prevs = Counter({k: v/sum(prevs.values()) for k, v in prevs.items()})
        for prev, curr in curr_given_prevs.items():
            curr_given_prevs[prev] = Counter({k: v/sum(curr.values()) for k, v in curr.items()})
        return prevs, curr_given_prevs

    @classmethod
    def sample_value(cls, col, op, db_path):
        if op == 'count':
            v = np.random.randint(1, 10)
            return str(v), [str(v)]
    
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
            raise Exception('Unknown op {}'.format(op))
        return str(val), [str(v) for v in val_toks]
 
    @classmethod
    def fill_template(cls, template, mapping, col_map, db_path):
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
        return column_mapped_query, value_mapped_query

    def sample_one(self, db_root, stats, allowed_db_ids, proc_cols, supports):
        p_prevs, p_curr_given_prevs = stats

        # sample templates
        templates = sorted(list(p_prevs.keys()))
        template_probs = [p_prevs[k] for k in templates]
        prev_template = np.random.choice(templates, p=template_probs)

        templates = sorted(list(p_curr_given_prevs[prev_template].keys()))
        template_probs = [p_curr_given_prevs[prev_template][k] for k in templates]
        curr_template = np.random.choice(templates, p=template_probs)
    
        ex = dict(prev_template=prev_template, p_prev_template=p_prevs[prev_template], curr_template=curr_template, p_curr_template=p_curr_given_prevs[prev_template][curr_template])
    
        # gather databases that can support this template
        vcols = {t for t in prev_template.split() + curr_template.split() if '_col_' in t}
        supported_dbs = []
        for db_id, supported_cols in supports.items():
            if db_id in allowed_db_ids and not vcols - supported_cols:
                # every col is supported
                supported_dbs.append(db_id)
        if not supported_dbs:
            raise NoSupportedDBException(curr_template)
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
        prev_column_mapped_query, prev_value_mapped_query = self.fill_template(prev_template, mapping, col_map, db_path)
        curr_column_mapped_query, curr_value_mapped_query = self.fill_template(curr_template, mapping, col_map, db_path)
    
        ex['recov'] = editsql_postprocess.postprocess_one(' '.join(curr_column_mapped_query), self.conv.database_schemas[db_id])
        ex['recov'] = re.sub('\s([0-9a-zA-Z_]+\.)\*\s', '', ex['recov'])
        ex['values'] = preprocess_nl2sql.SQLDataset.align_values(curr_column_mapped_query, curr_value_mapped_query)
    
        ex['prev_column_mapped'] = prev_column_mapped_query
        ex['prev_value_mapped'] = prev_value_mapped_query
        ex['curr_column_mapped'] = curr_column_mapped_query
        ex['curr_value_mapped'] = curr_value_mapped_query
        ex['norm_query'] = curr_column_mapped_query
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
        query['query_toks_no_value'] = editsql_postprocess.postprocess_one(' '.join(query['curr_column_mapped']), database_schemas[db_id]).replace('limit 1', 'limit_value').replace(' 1', ' value').replace('.', ' . ').split(' ')
        schema = evaluation.Schema(evaluation.get_schema(db_path))
        g_raw_res = timed_execute(db_path, query_recov, timeout=1, sleep=0.001, silent=silent)
        return g_raw_res
    
    def sample_executable_queries(self, db_root, num, db_ids, proc_cols, supports, stats, silent=False):
        queries = []
        pbar = tqdm.tqdm(desc='generate', total=num)
        while len(queries) < num:
            try:
                query = self.sample_one(db_root, stats, db_ids, proc_cols, supports)
            except NoSupportedDBException as e:
                pass
            except EmptyColumnException as e:
                pass
            except ValueAlignmentException as e:
                pass
            try:
                res = self.execute_query(db_root, query, self.conv.database_schemas, silent=silent)
            except Exception as e:
                print('failed to execute query: {}'.format(' '.join(query['curr_value_mapped'])))
                print(e)
                print()
                continue
            if res:
                queries.append(query)
                pbar.update(1)
        return queries

    def run_gen_on_split(self, size, SQLSampler, db_split, fout, args=None, save=True):
        args = args or self.args
        assert '.json' in fout
        fsql = fout.replace('.json', '.sql.pt')
        fques = fout.replace('.json', '.ques.pt')
        ffinal = fout.replace('.json', '.pt')
        if args.skip_consistency_check:
            ffinal = ffinal.replace('.pt', '.nocheck.pt')

        if os.path.isfile(fsql):
            print('loading {}'.format(fsql))
            batched_queries = torch.load(fsql)
        else:
            # sample some sql queries
            db_ids = sorted([k for k in self.conv.database_schemas.keys() if os.path.isfile(os.path.join(args.db, k, '{}.sqlite'.format(k)))])
            print(len(db_ids), 'total db ids found')
            assert db_ids
            train_dbs = set()
            with open(args.ftrain) as f:
                for ex in json.load(f):
                    if 'db_id' not in ex:
                        ex['db_id'] = ex['database_id']
                    train_dbs.add(ex['db_id'])

            proc_cols = SQLSampler.process_cols(db_ids, self.conv.database_schemas)
            supports = SQLSampler.process_supports(proc_cols)
            ftrain = getattr(args, 'ftrain', os.path.join('data', self.args.dataset, 'train.json'))
            stats = self.load_statistics(ftrain, SQLSampler, self.conv, proc_cols)

            if db_split == 'train':
                allowed_db_ids = [d for d in db_ids if d in train_dbs]
            else:
                allowed_db_ids = [d for d in db_ids if d not in train_dbs]

            queries = self.sample_executable_queries(args.db, size, allowed_db_ids, proc_cols, supports, stats, self.bert_tokenizer)

            batched_queries = dataset.Dataset()
            for i, ex in enumerate(queries):
                # encode tables
                try:
                    question_context, columns = preprocess.SQLDataset.build_contexts(ex['curr_column_mapped'], ex['prev_column_mapped'], ex['values'], self.conv.database_schemas[ex['db_id']], self.bert_tokenizer)
                except Exception as e:
                    print('Failed to build context')
                    print(e)
                    continue
                new = dict(
                    id='gen:{}'.format(i),
                    columns=columns,
                    db_id=ex['db_id'], 
                    query=ex['recov'],
                    g_query_norm=ex['curr_column_mapped'],
                    g_prev_query_norm=ex['prev_column_mapped'],
                    g_query_recov=ex['recov'],
                    g_values=ex['values'],
                    question_context=question_context,
                    cands_question=preprocess.SQLDataset.make_column_cands(question_context),
                )
                new['cands_question'] = preprocess.SQLDataset.make_cands(new, self.utt_vocab)
                batched_queries.append(new)

            if save:
                torch.save(batched_queries, fsql)

        if os.path.isfile(fques):
            print('loading {}'.format(fques))
            batched_queries = torch.load(fques)
        else:
            # generate some questions
            print('orig', len(batched_queries))
            batched_queries = batched_queries.keep(lambda ex: len(ex['question_context']) < 60)
            print('pruned', len(batched_queries))
            with torch.no_grad():
                preds_questions = self.run_pred(batched_queries, desc='generate questions', args=args)

            for ex in batched_queries:
                p = preds_questions[ex['id']]
                question_toks = p['utt_toks']

                # make original question
                question = self.bert_tokenizer.convert_tokens_to_string(question_toks)
                # casing
                for val_toks in ex['g_values']:
                    val = self.bert_tokenizer.convert_tokens_to_string(val_toks).strip(' \"\'`%')
                    question = question.replace(val.lower(), val)

                query_context = preprocess_nl2sql.SQLDataset.build_contexts(question_toks, ex['g_prev_query_norm'], self.conv.database_schemas[ex['db_id']], self.bert_tokenizer)
                ex['question'] = question
                ex['g_question_toks'] = question_toks
                ex['value_context'] = [self.bert_tokenizer.cls_token] + question_toks + [self.bert_tokenizer.sep_token]
                ex['query_context'] = query_context
                ex['cands_query'] = preprocess_nl2sql.SQLDataset.make_column_cands(query_context)
                osize = len(self.nl2sql.sql_vocab)
                ex['sup_query'] = preprocess_nl2sql.SQLDataset.make_sup_query(ex['g_query_norm'], ex['cands_query'], ex['g_values'], self.nl2sql.sql_vocab, self.bert_tokenizer)
                ex['cands_query'], ex['cands_value'] = preprocess_nl2sql.SQLDataset.make_cands(ex, self.nl2sql.sql_vocab)
                ex['pointer_query'], ex['pointer_value'] = preprocess_nl2sql.SQLDataset.make_query_pointer(ex['sup_query'], ex['cands_query'], ex['cands_value'], self.nl2sql.sql_vocab)
                nsize = len(self.nl2sql.sql_vocab)
                if nsize != osize:
                    raise Exception('vocab size increased!\n{}'.format(self.nl2sql.sql_vocab._index2word[osize:]))

            batch_queries = self.nl2sql.prune_train(batched_queries, self.nl2sql.args)
            if save:
                torch.save(batched_queries, fques)

        if os.path.isfile(ffinal):
            keep = torch.load(ffinal)
        else:
            if args.skip_consistency_check:
                keep = batched_queries
                for ex in keep:
                    # make original question
                    question = self.bert_tokenizer.convert_tokens_to_string(ex['g_question_toks'])
                    # casing
                    for val_toks in ex['g_values']:
                        val = self.bert_tokenizer.convert_tokens_to_string(val_toks).strip(' \"\'`%')
                        question = question.replace(val.lower(), val)
                    ex['question'] = question
            else:
                with torch.no_grad():
                    preds_queries = self.nl2sql.run_pred(batched_queries, args=args, desc='generate queries', force_non_interactive=True)
                results = self.nl2sql.compute_official_eval(batched_queries, preds_queries, return_every_result=True)
                # check for cycle consistency
                print({k: sum(v)/len(v) for k, v in results.items()})
                keep = []
                for ex, res in zip(batched_queries, results['official_ex']):
                    p = preds_queries[ex['id']]
                    question = ' '.join(ex['g_question_toks'])
                    if res:
                        keep.append(ex)
                    query = p['query']
                    # if p[
                    #     print('--- Sampled query')
                    #     print(ex['g_query'])
                    #     print('--- Question generation')
                    #     print(' '.join(ex['question_context']))
                    #     print(question)
                    #     print('--- Query generation')
                    #     print(p['query'])
                    #     print(res)
                    #     import pdb; pdb.set_trace()

                print('generated {} examples'.format(len(keep)))
                # assert 'best.pt' in args.resume or 'best.tar' in args.resume
            torch.save(keep, ffinal)

        debug = []
        for ex in keep:
            debug.append(dict(
                id=ex['id'],
                question=' '.join(ex['g_question_toks']),
                query=ex['query'],
                query_norm=' '.join(ex['g_query_norm']),
            ))
        with open(fout, 'wt') as f:
            json.dump(debug, f, indent=2)
        return keep
