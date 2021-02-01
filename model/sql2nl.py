import os
import json
import utils
import torch
import dataset
import preprocess_nl2sql
import editsql_preprocess
import preprocess_sql2nl as preprocess
import converter
from torch import nn
from torch.nn import functional as F
from model import rnn, decoder, rank_max
from model.model import Module as Base
from bleu import corpus_bleu, SmoothingFunction
from collections import defaultdict
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, get_linear_schedule_with_warmup


class Module(Base):

    @classmethod
    def load_nl2sql(cls, args):
        return Base.load_inst(args.fparser, dict(tables=args.tables, db=getattr(args, 'db', 'data/database'), dcache=args.dcache, beam_size=args.beam_size, batch=args.batch))

    def __init__(self, args, ext):
        super().__init__(args, ext)
        self.nl2sql = self.load_nl2sql(args)
        self.conv = converter.Converter(tables=getattr(args, 'tables', 'data/spider/tables'), db=getattr(args, 'db', 'data/database'))
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(args.dcache, cache_dir=args.dcache)
        self.utt_bert_embedder = DistilBertModel.from_pretrained(args.dcache, cache_dir=args.dcache)
        self.denc = 768
        self.demb = args.demb
        self.utt_vocab = ext['utt_voc']
        self.utt_emb = nn.Embedding.from_pretrained(ext['utt_emb'], freeze=False)
        self.pad_id = self.utt_vocab.word2index('PAD')

        self.dropout = nn.Dropout(args.dropout)
        self.bert_dropout = nn.Dropout(args.bert_dropout)

        self.utt_cand_trans = nn.LSTM(self.denc, self.demb//2, bidirectional=True, batch_first=True)
        self.utt_enc_trans = nn.LSTM(self.denc, args.drnn, bidirectional=True, batch_first=True)
        self.utt_pointer_decoder = decoder.PointerDecoder(demb=self.demb, denc=2*args.drnn, ddec=args.drnn, dropout=args.dec_dropout, num_layers=args.num_layers)

        if 'reranker' in ext:
            self.reranker = ext['reranker']
        else:
            self.reranker = rank_max.Module(args, ext, remove_invalid=False)

    @classmethod
    def get_num_train_steps(cls, train_size, batch_size, num_epoch):
        return train_size // batch_size * num_epoch

    def get_optimizer(self, train, lr=5e-5, warmup=0.1, weight_decay=0.01):
        num_total_steps = self.get_num_train_steps(len(train), self.args.batch, self.args.epoch)
        # remove pooler
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        # prepare optimizer and schedule (linear warmup and decay)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps * warmup, num_training_steps=num_total_steps)
        return optimizer, scheduler

    def better(self, metrics, best):
        return metrics['dev_official_em'] >= best.get('dev_official_em', -1)

    def featurize(self, batch):
        feat = defaultdict(list)
        cls_token = self.bert_tokenizer.cls_token
        sep_token = self.bert_tokenizer.sep_token
        for ex in batch:
            tens = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(ex['question_context']))
            feat['context'].append(tens)
            feat['context_mask'].append(torch.ones_like(tens))
            if self.training:
                feat['utt_pointer'].append(torch.tensor(ex['pointer_question']))
        feat['context'] = utils.pad_sequence(feat['context'], self.pad_id, self.device)
        feat['context_mask'] = utils.pad_sequence(feat['context_mask'], 0, self.device).float()
        feat['utt_pointer'] = utils.pad_sequence(feat['utt_pointer'], self.pad_id, self.device) if self.training else None
        feat['batch'] = batch
        return feat

    def forward(self, context, context_mask, utt_pointer, batch):
        B = len(batch)
        # reps for flattened columns
        bert = self.bert_dropout(self.utt_bert_embedder(context)[0])
        cand_trans, _ = rnn.run_rnn(self.utt_cand_trans, bert, context_mask.sum(1).long())
        enc_trans, _ = rnn.run_rnn(self.utt_enc_trans, bert, context_mask.sum(1).long())
        cand = torch.cat([
            self.utt_emb.weight.unsqueeze(0).repeat(B, 1, 1),
            cand_trans,
        ], dim=1)
        cand_mask = torch.cat([
            torch.ones(B, len(self.utt_vocab)).float().to(self.device),
            context_mask,
        ], dim=1)

        if not self.should_beam_search():
            utt_dec = self.utt_pointer_decoder(
                emb=self.dropout(cand), emb_mask=cand_mask,
                enc=self.dropout(enc_trans), enc_mask=context_mask.float(),
                state0=None,
                gt=utt_pointer if self.training else None,
                max_len=self.args.max_query_len,
                batch=batch,
            )
        else:
            utt_dec = self.utt_pointer_decoder.beam_search(
                emb=self.dropout(cand), emb_mask=cand_mask,
                enc=self.dropout(enc_trans), enc_mask=context_mask.float(),
                eos_ind=self.utt_vocab.word2index('EOS'),
                max_len=self.args.max_query_len,
                batch=batch,
                beam_size=self.args.beam_size,
            )
        return dict(utt_dec=utt_dec)

    def should_beam_search(self):
        return (not self.training) and self.args.beam_size > 0

    def score(self, context, context_mask, utt_pointer, batch):
        B = len(batch)
        # reps for flattened columns
        bert = self.bert_dropout(self.utt_bert_embedder(context)[0])
        cand_trans, _ = rnn.run_rnn(self.utt_cand_trans, bert, context_mask.sum(1).long())
        enc_trans, _ = rnn.run_rnn(self.utt_enc_trans, bert, context_mask.sum(1).long())
        cand = torch.cat([
            self.utt_emb.weight.unsqueeze(0).repeat(B, 1, 1),
            cand_trans,
        ], dim=1)
        cand_mask = torch.cat([
            torch.ones(B, len(self.utt_vocab)).float().to(self.device),
            context_mask,
        ], dim=1)

        utt_dec = self.utt_pointer_decoder.forward(
            emb=self.dropout(cand), emb_mask=cand_mask,
            enc=self.dropout(enc_trans), enc_mask=context_mask.float(),
            state0=None,
            gt=utt_pointer,
            max_len=self.args.max_query_len,
            batch=batch,
        )
        normed = torch.log_softmax(utt_dec, dim=2)
        eos = self.utt_vocab.word2index('EOS')
        scores = []
        for score, inds, ex in zip(normed, utt_pointer, batch):
            valid = inds.tolist()
            if eos in valid:
                valid = valid[:valid.index(eos)+1]

            greedy = score.max(1)[1].tolist()
            if eos in greedy:
                greedy = greedy[:greedy.index(eos)+1]
            greedy_score = score.max(1)[0].sum()
            score_sum = sum([score[i, j].item() for i, j in enumerate(valid)]) / len(valid)
            scores.append(score_sum)
        return scores

    def extract_preds(self, out, feat, batch):
        preds = {}
        for utt_dec, ex in zip(out['utt_dec'], batch):
            if self.should_beam_search():
                for beam in utt_dec:
                    beam.toks = preprocess.SQLDataset.recover_question(beam.inds, ex['cands_question'], voc=self.utt_vocab)
                b = self.reranker.rerank(utt_dec, ex)
                inds = b.inds
                toks = b.toks
            else:
                inds = utt_dec.max(1)[1].tolist()
                toks = preprocess.SQLDataset.recover_question(inds, ex['cands_question'], voc=self.utt_vocab)

            utt = self.bert_tokenizer.convert_tokens_to_string(toks)
            for c in ex['columns']:
                if ' id' in c['name']:
                    utt = utt.replace(c['name'], c['name'].replace(' id', ''))
            preds[ex['id']] = dict(
                utt_pointer=inds,
                utt_toks=toks,
                utt=utt,
                raw_scores=utt_dec.to(torch.device('cpu')) if not self.should_beam_search() else None,
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
            query_context = preprocess_nl2sql.SQLDataset.build_contexts(question_toks, db, self.bert_tokenizer)

            if 'g_sql' not in ex:
                ex['g_sql'] = self.conv.build_sql(ex['query'], db_id)

            new = dict(
                id=ex['id'],
                question=ex['question'],
                db_id=db_id, 
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
                preds = self.nl2sql.run_pred(generated, self.nl2sql.args, verbose=True, desc='cycle_pred')
            metrics.update(self.nl2sql.compute_official_eval(generated, preds))
        return metrics

    def compute_loss(self, out, feat, batch):
        losses = {}
        dec = out['utt_dec']
        gold = feat['utt_pointer']
        losses['sup'] = F.cross_entropy(dec.view(gold.numel(), -1), gold.view(-1), ignore_index=self.pad_id)
        return losses

    def get_debug(self, ex, pred):
        return dict(
            gold_query=ex['query'],
            utt=pred['utt'],
            gold_utt=self.bert_tokenizer.convert_tokens_to_string(ex['g_question_toks']),
            context=self.bert_tokenizer.convert_tokens_to_string(ex['question_context']),
        )

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
    def prune_train(cls, train, args, limit=1000):
        return train

    @classmethod
    def prune_dev(cls, dev, args):
        return dev

    @classmethod
    def get_stats(cls, splits, ext):
        stats = {}
        all_data = []
        for k, v in splits.items():
            stats[k] = len(v)
            all_data.extend(v)
        question_context_len = [len(ex['question_context']) for ex in all_data]
        question_len = [len(ex['pointer_question']) for ex in splits['train']]
        stats.update({
            'utt': ext['utt_voc'],
            'max_question_context': max(question_context_len),
            'max_question': max(question_len),
        })
        return stats

    def run_gen_on_split(self, size, SQLSampler, db_split, fout, args=None, save=True):
        args = args or self.args
        assert '.json' in fout
        fsql = fout.replace('.json', '.sql.pt')
        fques = fout.replace('.json', '.ques.pt')
        ffinal = fout.replace('.json', '.pt')

        self.conv = self.nl2sql.conv = converter.Converter(tables=getattr(args, 'tables', 'data/spider/tables'), db=getattr(args, 'db', 'data/database'))

        # print(fsql, os.path.isfile(fsql))
        # raise
        if os.path.isfile(fsql):
            print('loading {}'.format(fsql))
            batched_queries = torch.load(fsql)
        else:
            # sample some sql queries

            # compute training template distribution
            # compute db supports
            schema_tokens, column_names, database_schemas = editsql_preprocess.read_database_schema(args.tables, {}, {}, {})
            db_ids = sorted(list(database_schemas.keys()))
            train_dbs = set()
            with open(args.ftrain) as f:
                for ex in json.load(f):
                    train_dbs.add(ex['db_id'])

            proc_cols = SQLSampler.process_cols(db_ids, database_schemas)
            supports = SQLSampler.process_supports(proc_cols)
            stats = SQLSampler.load_statistics(args.ftrain, column_names, schema_tokens, database_schemas, proc_cols)

            if db_split == 'train':
                allowed_db_ids = [d for d in db_ids if d in train_dbs]
            else:
                allowed_db_ids = [d for d in db_ids if d not in train_dbs]

            queries = SQLSampler.sample_executable_queries(args.db, size, allowed_db_ids, schema_tokens, column_names, database_schemas, proc_cols, supports, stats, self.bert_tokenizer, args)

            batched_queries = dataset.Dataset()
            for i, ex in enumerate(queries):
                # encode tables
                try:
                    question_context, columns = preprocess.SQLDataset.build_contexts(ex['column_mapped'], ex['values'], self.conv.database_schemas[ex['db_id']], self.bert_tokenizer)
                except Exception as e:
                    print('Failed to build context')
                    print(e)
                    continue
                new = dict(
                    id='gen:{}'.format(i),
                    columns=columns,
                    db_id=ex['db_id'], 
                    query=ex['recov'],
                    g_query_norm=ex['column_mapped'],
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
                if args.beam_size > 0:
                    self.beam_size = args.beam_size
                    self.reranker = rank_max.Module(self.args, self.ext, remove_invalid=False)
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

                query_context = preprocess_nl2sql.SQLDataset.build_contexts(question_toks, self.conv.database_schemas[ex['db_id']], self.bert_tokenizer)
                ex['question'] = question
                ex['g_question_toks'] = question_toks
                ex['value_context'] = [self.bert_tokenizer.cls_token] + question_toks + [self.bert_tokenizer.sep_token]
                ex['query_context'] = query_context
                ex['cands_query'] = preprocess_nl2sql.SQLDataset.make_column_cands(query_context)
                osize = len(self.nl2sql.sql_vocab)
                ex['sup_query'] = preprocess_nl2sql.SQLDataset.make_sup_query(ex['g_query_norm'], ex['cands_query'], ex['g_values'], self.nl2sql.sql_vocab, self.bert_tokenizer, train=False)
                ex['cands_query'], ex['cands_value'] = preprocess_nl2sql.SQLDataset.make_cands(ex, self.nl2sql.sql_vocab)
                ex['pointer_query'], ex['pointer_value'] = preprocess_nl2sql.SQLDataset.make_query_pointer(ex['sup_query'], ex['cands_query'], ex['cands_value'], self.nl2sql.sql_vocab)
                nsize = len(self.nl2sql.sql_vocab)
                if nsize != osize:
                    raise Exception('vocab size increased!\n{}'.format(self.nl2sql.sql_vocab._index2word[osize:]))
            batch_queries = self.nl2sql.prune_train(batched_queries, self.nl2sql.args)
            if save:
                torch.save(batched_queries, fques)

        if os.path.isfile(ffinal) and False:
            keep = torch.load(ffinal)
        else:
            if args.skip_consistency_check:
                keep = batched_queries
            else:
                with torch.no_grad():
                    if args.beam_size > 0:
                        self.nl2sql.beam_size = args.beam_size
                        self.nl2sql.reranker = rank_max.Module(self.nl2sql.args, self.nl2sql.ext, remove_invalid=True)
                    preds_queries = self.nl2sql.run_pred(batched_queries, args=args, desc='generate queries')
 
                results = self.nl2sql.compute_official_eval(batched_queries, preds_queries, return_every_result=True)
                # check for cycle consistency
                print({k: sum(v)/len(v) for k, v in results.items()})
                keep = []
                debug = []
                for ex, res in zip(batched_queries, results['official_ex']):
                    p = preds_queries[ex['id']]
                    question = ' '.join(ex['g_question_toks'])
                    if res:
                        keep.append(ex)
                    debug.append(dict(ex=ex, res=res))
                    query = p['query']
                    # if p['value_toks']:
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
                assert '.pt' in args.resume or '.tar' in args.resume
                torch.save(debug, ffinal.replace('.pt', '.debug.pt'))
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
