import os

import re
import tqdm
import copy
import utils
import torch
import converter
import timed_execute
import preprocess_nl2sql as preprocess
from eval_scripts import evaluation
from torch import nn
from torch.nn import functional as F
from model import rnn, decoder, rank_max
from model.model import Module as Base
from bleu import corpus_bleu, SmoothingFunction
from collections import defaultdict
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, get_linear_schedule_with_warmup


class Module(Base):

    def __init__(self, args, ext):
        super().__init__(args, ext)
        self.conv = converter.Converter(tables=getattr(args, 'tables', 'data/spider/tables'), db=getattr(args, 'db', 'data/database'))
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(args.dcache + '/vocab.txt', cache_dir=args.dcache)
        self.bert_embedder = DistilBertModel.from_pretrained(args.dcache, cache_dir=args.dcache)
        self.value_bert_embedder = DistilBertModel.from_pretrained(args.dcache, cache_dir=args.dcache)
        self.denc = 768
        self.demb = args.demb
        self.sql_vocab = ext['sql_voc']
        self.sql_emb = nn.Embedding.from_pretrained(ext['sql_emb'], freeze=False)
        self.pad_id = self.sql_vocab.word2index('PAD')

        self.dropout = nn.Dropout(args.dropout)
        self.bert_dropout = nn.Dropout(args.bert_dropout)
        self.table_sa_scorer = nn.Linear(self.denc, 1)
        self.col_sa_scorer = nn.Linear(self.denc, 1)
        self.col_trans = nn.LSTM(self.denc, self.demb//2, bidirectional=True, batch_first=True)
        self.table_trans = nn.LSTM(self.denc, args.drnn, bidirectional=True, batch_first=True)
        self.pointer_decoder = decoder.PointerDecoder(demb=self.demb, denc=2*args.drnn, ddec=args.drnn, dropout=args.dec_dropout, num_layers=args.num_layers)

        self.utt_trans = nn.LSTM(self.denc, self.demb//2, bidirectional=True, batch_first=True)
        self.value_decoder = decoder.PointerDecoder(demb=self.demb, denc=self.denc, ddec=args.drnn, dropout=args.dec_dropout, num_layers=args.num_layers)

        self.evaluator = evaluation.Evaluator()
        if 'reranker' in ext:
            self.reranker = ext['reranker']
        else:
            self.reranker = rank_max.Module(args, ext, remove_invalid=True)

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
            if self.training:
                feat['query_pointer'].append(torch.tensor(ex['pointer_query']))
                feat['value_pointer'].append(torch.tensor(ex['pointer_value']))

            feat['utterance'].append(torch.tensor(self.bert_tokenizer.convert_tokens_to_ids([cls_token] + ex['g_question_toks'] + [sep_token])))
            tables = []
            tables_mask = []
            starts, ends = [], []
            for t in ex['query_context']:
                tens = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(t['toks']))
                tables.append(tens)
                tables_mask.append(torch.ones_like(tens))
                starts.append([c['start'] for c in t['columns']])
                ends.append([c['end'] for c in t['columns']])
            feat['tables'].append(utils.pad_sequence(tables, self.bert_tokenizer.pad_token_id, self.device))
            feat['tables_mask'].append(utils.pad_sequence(tables_mask, 0, self.device).float())
            feat['starts'].append(starts)
            feat['ends'].append(ends)

        feat['query_pointer'] = utils.pad_sequence(feat['query_pointer'], self.pad_id, self.device) if self.training else None
        feat['value_pointer'] = utils.pad_sequence(feat['value_pointer'], self.pad_id, self.device) if self.training else None

        feat['utterance_mask'] = utils.pad_sequence([torch.ones(len(t)) for t in feat['utterance']], 0, self.device)
        feat['utterance'] = utils.pad_sequence(feat['utterance'], self.pad_id, self.device)

        feat['batch'] = batch
        return feat

    def forward(self, utterance, utterance_mask, tables, tables_mask, starts, ends, query_pointer, value_pointer, batch):
        B = len(batch)

        # reps for flattened columns
        col_reps = []
        col_mask = []
        # reps for each table
        table_reps = []
        table_mask = []
        for ids_table, mask_table, start_table, end_table in zip(tables, tables_mask, starts, ends):
            bert_table = self.bert_dropout(self.bert_embedder(ids_table)[0])
            table_col_reps = []
            for bert_col, start_col, end_col in zip(bert_table, start_table, end_table):
                cols = [bert_col[cs:ce] for cs, ce in zip(start_col, end_col)]
                mask = [torch.ones(len(e)) for e in cols]
                pad = nn.utils.rnn.pad_sequence(cols, batch_first=True, padding_value=0)
                mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0).float().to(self.device)
                # compute selfattn for this column
                scores = self.col_sa_scorer(pad).squeeze(2)
                normalized_scores = F.softmax(scores - (1-mask)*1e20, dim=1)
                col_sa = pad.mul(normalized_scores.unsqueeze(2).expand_as(pad)).sum(1)
                table_col_reps.append(col_sa)
            table_col_reps = torch.cat(table_col_reps, dim=0)
            col_reps.append(table_col_reps)
            col_mask.append(torch.ones(len(table_col_reps)))

            # compute selfattn for this talbe
            scores = self.table_sa_scorer(bert_table).squeeze(2)
            normalized_scores = F.softmax(scores - (1-mask_table)*1e20, dim=1)
            tab_sa = bert_table.mul(normalized_scores.unsqueeze(2).expand_as(bert_table)).sum(1)
            table_reps.append(tab_sa)
            table_mask.append(torch.ones(len(tab_sa)))

        col_reps = nn.utils.rnn.pad_sequence(col_reps, batch_first=True, padding_value=0)
        col_mask = nn.utils.rnn.pad_sequence(col_mask, batch_first=True, padding_value=0).to(self.device)
        table_reps = nn.utils.rnn.pad_sequence(table_reps, batch_first=True, padding_value=0)
        table_mask = nn.utils.rnn.pad_sequence(table_mask, batch_first=True, padding_value=0).to(self.device)

        col_trans, _ = rnn.run_rnn(self.col_trans, col_reps, col_mask.sum(1).long())
        table_trans, _ = rnn.run_rnn(self.table_trans, table_reps, table_mask.sum(1).long())
        table_trans = self.dropout(table_trans)

        cand = self.dropout(torch.cat([
            self.sql_emb.weight.unsqueeze(0).repeat(B, 1, 1),
            col_trans,
        ], dim=1))
        cand_mask = torch.cat([
            torch.ones(B, len(self.sql_vocab)).float().to(self.device),
            col_mask
        ], dim=1)

        if not self.should_beam_search():
            query_dec = self.pointer_decoder(
                emb=cand, emb_mask=cand_mask,
                enc=table_trans, enc_mask=table_mask.float(),
                state0=None,
                gt=query_pointer,
                max_len=self.args.max_query_len,
                batch=batch,
            )
        else:
            query_dec = self.pointer_decoder.beam_search(
                emb=cand, emb_mask=cand_mask,
                enc=table_trans, enc_mask=table_mask.float(),
                eos_ind=self.sql_vocab.word2index('EOS'),
                max_len=self.args.max_query_len,
                batch=batch,
                beam_size=self.args.beam_size,
            )

        utt = self.bert_dropout(self.value_bert_embedder(utterance)[0])
        utt_trans, _ = rnn.run_rnn(self.utt_trans, utt, utterance_mask.sum(1).long())
        cand = self.dropout(torch.cat([
            self.sql_emb.weight.unsqueeze(0).repeat(B, 1, 1),
            utt_trans,
        ], dim=1))
        cand_mask = torch.cat([
            torch.ones(B, len(self.sql_vocab)).float().to(self.device),
            utterance_mask
        ], dim=1)

        value_dec = self.value_decoder(
            emb=cand, emb_mask=cand_mask,
            enc=utt, enc_mask=utterance_mask.float(),
            state0=None,
            gt=value_pointer if self.training else None,
            max_len=self.args.max_value_len,
            batch=batch,
        )
        return dict(query_dec=query_dec, value_dec=value_dec)

    def should_beam_search(self):
        return (not self.training) and self.args.beam_size > 0

    def recover_values(self, value_toks):
        values = '___'.join(value_toks).split('SEP')
        proc = []
        for v in values:
            if v:
                bert = self.bert_tokenizer.convert_tokens_to_string(v.split('___'))
                try:
                    float(bert.replace(' ', ''))
                except:
                    proc.append(bert)
                else:
                    proc.append(bert.replace(' ', ''))
        return proc

    def extract_query(self, pointer, value_pointer, ex):
        toks, value_toks = preprocess.SQLDataset.recover_query(pointer, ex['cands_query'], value_pointer, ex['cands_value'], voc=self.sql_vocab)
        post = toks[:]
        values = self.recover_values(value_toks)

        q_low = ex['question'].lower()
        for i, v in enumerate(values):
            # mark = v.strip(' ' + string.punctuation).lower()
            mark = v.strip(' \"\'`%').lower()
            if mark in q_low:
                start = q_low.index(mark)
                values[i] = v.replace(mark, ex['question'][start:start+len(mark)])

        try:
            post = post_no_value = self.conv.recover(' '.join(post), ex['db_id'])
            post = post_no_value = re.sub('\s([0-9a-zA-Z_]+\.)\*\s', '', post)
            if self.args.keep_values:
                post = self.postprocess_value(post, values)
        except Exception as e:
            post = post_no_value = repr(e)

        # fix spacing
        spacing = [
            ('` ` ', '"'), ('` `', '"'), ("''", '"'),
            ('> =', '>='), ('< =', '<='), ('! =', '!='),
            ("'% ", "'%"), (" %'", "%'"),
        ]
        for f, t in spacing:
            post = post.replace(f, t)
        return post, post_no_value, toks, value_toks, values

    def extract_preds(self, out, feat, batch):
        preds = {}
        for pointer, value_pointer, ex in zip(out['query_dec'], out['value_dec'].max(2)[1].tolist(), batch):
            if self.should_beam_search():
                for beam in pointer:
                    beam.post, beam.post_no_value, beam.toks, beam.value_toks, beam.values = self.extract_query(beam.inds, value_pointer, ex)
                b = self.reranker.rerank(pointer, ex)
                post, post_no_value, toks, value_toks, values = b.post, b.post_no_value, b.toks, b.value_toks, b.values
            else:
                pointer = pointer.max(1)[1].tolist()
                post, post_no_value, toks, value_toks, values = self.extract_query(pointer, value_pointer, ex)

            preds[ex['id']] = dict(
                query_pointer=pointer,
                query_toks=toks,
                query_no_value=post_no_value,
                query=post,
                value_pointer=value_pointer,
                value_toks=value_toks,
                values=values,
            )
        return preds

    def compute_official_eval(self, dev, dev_preds, return_every_result=False, allow_None_gold_execute=True):
        results = dict(official_em=[], official_ex=[])
        for ex in tqdm.tqdm(dev, desc='official eval'):
            p = dev_preds[ex['id']]
            g_str = ex['query']
            db_name = ex['db_id']
            db = os.path.join(getattr(self.args, 'db', 'data/database'), db_name, db_name + ".sqlite")
            p_str = dev_preds[ex['id']]['query']

            if 'g_sql' not in ex:
                ex['g_sql'] = self.conv.build_sql(ex['query'], ex['db_id'])

            g_sql = ex['g_sql']
            # the offical eval script is buggy and modifies arguments in place
            try:
                p_sql = self.conv.build_sql(p_str, ex['db_id'])
                em = self.evaluator.eval_exact_match(copy.deepcopy(p_sql), copy.deepcopy(g_sql))
            except Exception as e:
                p_sql = None
                em = False
            # if not em:
            #     print(g_str)
            #     print(p_str)
            #     print(ex['final_sql_parse'])
            #     import pdb; pdb.set_trace()
            results['official_em'].append(em)

            if self.args.keep_values:
                if 'g_ex' not in ex:  # cache result
                    ex['g_ex'] = timed_execute.timed_execute(db, g_str, query_sql=copy.deepcopy(g_sql), timeout=3, sleep=0.001)
                    # print(g_str.encode("utf-8", errors='ignore'))
                    # print(repr(g_sql).encode("utf-8", errors='ignore'))
                    # print('cached result')
                    # print(repr(ex['g_ex']).encode("utf-8", errors='ignore'))
                    # print()
                g_ex = ex['g_ex']
                if p_sql is None:
                    p_ex = None
                else:
                    p_ex = timed_execute.timed_execute(db, p_str, query_sql=copy.deepcopy(p_sql), timeout=3, silent=True, sleep=0.001)
                if allow_None_gold_execute and not g_ex:
                    exe = 0
                else:
                    exe = 0 if p_ex is None else p_ex == g_ex
                results['official_ex'].append(exe)

        metrics = {k: sum(v) / len(dev) for k, v in results.items()}
        return results if return_every_result else metrics

    def compute_metrics(self, data, preds):
        em = 0
        hyps, refs = [], []
        for ex in data:
            if ex['id'] not in preds and self.training:
                continue
            p = preds[ex['id']]
            gsql = ex['query'].lower().split()
            psql = p['query'].lower().split()
            refs.append([gsql])
            hyps.append(psql)
            em += psql == gsql
        metrics = {
            'em': em/len(data),
            'bleu': corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method3),
        }
        if not self.training:
            metrics.update(self.compute_official_eval(data, preds))
        else:
            metrics['official_em'] = metrics['em']
        return metrics

    def compute_loss(self, out, feat, batch):
        dec = out['query_dec']
        gold = feat['query_pointer']
        query = F.cross_entropy(dec.view(gold.numel(), -1), gold.view(-1), ignore_index=self.pad_id)
        losses =  dict(query=query)

        if self.args.keep_values:
            dec = out['value_dec']
            gold = feat['value_pointer']
            losses['value'] = F.cross_entropy(dec.view(gold.numel(), -1), gold.view(-1), ignore_index=self.pad_id)
        return losses

    def get_debug(self, ex, pred):
        gold_query = ex['query']
        pred_query = pred['query']
        return dict(
            question=' '.join(ex['g_question_toks']),
            query=pred_query,
            gold_query=gold_query,
            match=pred_query == gold_query,
        )

    @classmethod
    def postprocess_value(cls, post, values):
        i = 0
        while i < len(values):
            v = values[i]
            m = re.search(r'\s+1(?:\s+|$)', post)
            if m is None:
                break
            post = post[:m.start()] + m.group().replace('1', v.replace("' ", "'").replace(" '", "'")) + post[m.end():]
            post = post.replace('  ', ' ')
            i += 1
        return post

    def compute_upperbound(self, dev):
        preds = {}
        for ex in dev:
            toks, value_toks = preprocess.SQLDataset.recover_query(ex['pointer_query'], ex['cands_query'], ex['pointer_value'], ex['cands_value'], voc=self.sql_vocab)
            post = post0 = self.conv.recover(' '.join(toks), ex['db_id'])
            values = self.recover_values(value_toks)
            q_low = ex['question'].lower()
            for i, v in enumerate(values):
                mark = v.strip(' \"\'`%').lower()
                if mark in q_low:
                    start = q_low.index(mark)
                    values[i] = v.replace(mark, ex['question'][start:start+len(mark)])

            # apply fix and casing
            for i, v in enumerate(values):
                words = v.split()
                fixed = [preprocess.value_replace.get(w, w) for w in words]
                values[i] = ' '.join(fixed)

            if self.args.keep_values:
                post = post1 = self.postprocess_value(post, values)

            # fix spacing
            spacing = [
                ('` ` ', '"'), ("''", '"'),
                ('> =', '>='), ('< =', '<='),
                ("'% ", "'%"), (" %'", "%'"),
            ]
            for f, t in spacing:
                post = post.replace(f, t)

            preds[ex['id']] = dict(
                query_pointer=ex['pointer_query'],
                query_toks=toks,
                query=post,
                value_toks=value_toks,
                values=values,
            )
        self.eval()
        metrics = self.compute_metrics(dev, preds)
        return metrics

    @classmethod
    def prune_train(cls, train, args, limit=1000):
        orig = len(train)
        train = train.keep(lambda ex: sum(len(t['toks']) for t in ex['query_context']) < limit)
        print('pruned length from {} to {}'.format(orig, len(train)))
        orig = len(train)
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
        query_context_len = [sum(len(t['toks']) for t in ex['query_context']) for ex in all_data]
        question_len = [len(ex['value_context']) for ex in all_data]
        query_len = [len(ex['pointer_query']) for ex in splits['train']]
        value_len = [len(ex['pointer_value']) for ex in splits['train']]
        stats.update({
            'sql': ext['sql_voc'],
            'max_query_context': max(query_context_len),
            'max_question': max(question_len),
            'max_query_len': max(query_len),
            'max_value_len': max(value_len),
        })
        return stats
