import os
import re
import tqdm
import copy
import json
import torch
import string
import random
import sqlite3
import preprocess
import subprocess
import editsql_postprocess
import numpy as np
import embeddings as E
from eval_scripts import evaluation
from vocab import Vocab
from torch import nn
from torch.nn import functional as F
from model import attention, rnn
from model.model import Module as Base
from bleu import corpus_bleu, SmoothingFunction
from collections import defaultdict, Counter
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, WarmupLinearSchedule


BERT_NAME = 'distilbert-base-uncased'
SQL_PRIMITIVES = {'select', 'from', 'not', 'in', 'where', 'max', 'min', 'avg'}


def pad_sequence(inds, pad, device=None):
    out = nn.utils.rnn.pad_sequence(inds, batch_first=True, padding_value=pad)
    return out if device is None else out.to(device)


class PointerDecoder(nn.Module):

    def __init__(self, demb, denc, ddec, dropout=0, num_layers=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(demb, ddec, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.attn = attention.DotAttn(dropout=dropout, dec_trans=nn.Linear(ddec, denc))
        self.go = nn.Parameter(torch.Tensor(demb))
        self.feat_lin = nn.Linear(denc+ddec, demb+1)
        nn.init.uniform_(self.go, -0.1, 0.1)

    def forward(self, emb, emb_mask, enc, enc_mask, state0=None, gt=None, max_len=50, batch=None):
        B, n_candidates, _ = emb.size()
        emb_t = self.go.unsqueeze(0).repeat(B, 1)
        picked_t = torch.zeros(B, n_candidates).to(emb.device)
        state_t = state0
        scores = []
        for t in range(gt.size(1) if gt is not None else max_len):
            emb_t = self.dropout(emb_t)
            h_t, state_t = self.rnn(emb_t.unsqueeze(1), state_t)
            h_t = self.dropout(h_t)
            attn_t, _ = self.attn(enc, enc_mask, h_t)
            attn_t = self.dropout(attn_t)
            feat_t = self.feat_lin(torch.cat([h_t, attn_t], dim=2))

            cand_t = torch.cat([emb, picked_t.unsqueeze(2)], dim=2)
            score_t = feat_t.expand_as(cand_t).mul(cand_t).sum(2)
            score_t = score_t - (1-emb_mask)*1e20
            max_t = gt[:, t] if gt is not None else score_t.max(1)[1]
            for i, (p, c) in enumerate(zip(picked_t, max_t)):
                p[c].fill_(1)
            emb_t = torch.stack([emb[i, j] for i, j in enumerate(max_t.tolist())], dim=0)
            scores.append(score_t)
        return torch.stack(scores, dim=1)


class Module(Base):

    def __init__(self, args, ext):
        super().__init__(args, ext)
        self.database_schemas = ext['database_schemas']
        self.database_content = ext['db_content']
        self.kmaps = ext['kmaps']
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_NAME, cache_dir=args.dcache)
        self.bert_embedder = DistilBertModel.from_pretrained(BERT_NAME, cache_dir=args.dcache)
        self.value_bert_embedder = DistilBertModel.from_pretrained(BERT_NAME, cache_dir=args.dcache)
        self.utt_bert_embedder = DistilBertModel.from_pretrained(BERT_NAME, cache_dir=args.dcache)
        self.denc = 768
        self.demb = args.demb
        self.sql_vocab = ext['sql_voc']
        self.utt_vocab = ext['utt_voc']
        self.sql_emb = nn.Embedding.from_pretrained(ext['sql_emb'], freeze=False)
        self.utt_emb = nn.Embedding.from_pretrained(ext['utt_emb'], freeze=False)
        self.pad_id = self.sql_vocab.word2index('PAD')

        self.dropout = nn.Dropout(args.dropout)
        self.bert_dropout = nn.Dropout(args.bert_dropout)
        self.table_sa_scorer = nn.Linear(self.denc, 1)
        self.col_sa_scorer = nn.Linear(self.denc, 1)
        self.col_trans = nn.LSTM(self.denc, self.demb//2, bidirectional=True, batch_first=True)
        self.table_trans = nn.LSTM(self.denc, args.drnn, bidirectional=True, batch_first=True)
        self.pointer_decoder = PointerDecoder(demb=self.demb, denc=2*args.drnn, ddec=args.drnn, dropout=args.dec_dropout, num_layers=args.num_layers)

        self.utt_trans = nn.LSTM(self.denc, self.demb//2, bidirectional=True, batch_first=True)
        self.value_decoder = PointerDecoder(demb=self.demb, denc=self.denc, ddec=args.drnn, dropout=args.dec_dropout, num_layers=args.num_layers)

        self.utt_table_sa_scorer = nn.Linear(self.denc, 1)
        self.utt_col_sa_scorer = nn.Linear(self.denc, 1)
        self.utt_col_trans = nn.LSTM(self.denc, self.demb//2, bidirectional=True, batch_first=True)
        self.utt_table_trans = nn.LSTM(self.denc, args.drnn, bidirectional=True, batch_first=True)
        self.utt_pointer_decoder = PointerDecoder(demb=self.demb, denc=2*args.drnn, ddec=args.drnn, dropout=args.dec_dropout, num_layers=args.num_layers)

        self.evaluator = evaluation.Evaluator()

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
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_total_steps * warmup, t_total=num_total_steps)
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
                feat['utt_pointer'].append(torch.tensor(ex['pointer_question']))

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
            feat['tables'].append(pad_sequence(tables, self.bert_tokenizer.pad_token_id, self.device))
            feat['tables_mask'].append(pad_sequence(tables_mask, 0, self.device).float())
            feat['starts'].append(starts)
            feat['ends'].append(ends)

            tables = []
            tables_mask = []
            starts, ends = [], []
            for t in ex['question_context']:
                tens = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(t['toks']))
                tables.append(tens)
                tables_mask.append(torch.ones_like(tens))
                starts.append([c['start'] for c in t['columns']])
                ends.append([c['end'] for c in t['columns']])
            feat['utt_tables'].append(pad_sequence(tables, self.bert_tokenizer.pad_token_id, self.device))
            feat['utt_tables_mask'].append(pad_sequence(tables_mask, 0, self.device).float())
            feat['utt_starts'].append(starts)
            feat['utt_ends'].append(ends)

        feat['query_pointer'] = pad_sequence(feat['query_pointer'], self.pad_id, self.device) if self.training else None
        feat['value_pointer'] = pad_sequence(feat['value_pointer'], self.pad_id, self.device) if self.training else None
        feat['utt_pointer'] = pad_sequence(feat['utt_pointer'], self.pad_id, self.device) if self.training else None

        feat['utterance_mask'] = pad_sequence([torch.ones(len(t)) for t in feat['utterance']], 0, self.device)
        feat['utterance'] = pad_sequence(feat['utterance'], self.pad_id, self.device)

        feat['batch'] = batch
        return feat

    def utt_to_sql(self, utterance, ex, return_inp=False):
        cp = copy.deepcopy(ex)
        toks = [self.bert_tokenizer.cls_token] + self.bert_tokenizer.tokenize(utterance) + [self.bert_tokenizer.sep_token]
        ids = self.bert_tokenizer.convert_tokens_to_ids(toks)
        cp['utterance'] = dict(gloss=utterance, toks=toks, tens=torch.tensor(ids))
        self.make_context(cp, self.database_schemas[cp['db_id']], self.bert_tokenizer)
        self.make_pointer(cp, self.sql_vocab, self.utt_vocab, self.bert_tokenizer)
        batch = [cp]
        feat = self.featurize(batch)
        out = self.forward(**feat)
        preds = list(self.extract_preds(out, feat, batch).values())[0]
        if return_inp:
            return preds, cp
        else:
            return preds['query']

    def forward(self, utterance, utterance_mask, tables, tables_mask, starts, ends, query_pointer, value_pointer, utt_tables, utt_tables_mask, utt_starts, utt_ends, utt_pointer, batch):
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

        query_dec = self.pointer_decoder(
            emb=cand, emb_mask=cand_mask,
            enc=table_trans, enc_mask=table_mask.float(),
            state0=None,
            gt=query_pointer if self.training else None,
            max_len=self.args.max_query_len,
            batch=batch,
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

        # reps for each table
        # reps for flattened columns
        col_reps = []
        col_mask = []
        # reps for each table
        table_reps = []
        table_mask = []
        for ids_table, mask_table, start_table, end_table in zip(utt_tables, utt_tables_mask, utt_starts, utt_ends):
            bert_table = self.bert_dropout(self.utt_bert_embedder(ids_table)[0])
            table_col_reps = []
            for bert_col, start_col, end_col in zip(bert_table, start_table, end_table):
                cols = [bert_col[cs:ce] for cs, ce in zip(start_col, end_col)]
                mask = [torch.ones(len(e)) for e in cols]
                pad = nn.utils.rnn.pad_sequence(cols, batch_first=True, padding_value=0)
                mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0).float().to(self.device)
                # compute selfattn for this column
                scores = self.utt_col_sa_scorer(pad).squeeze(2)
                normalized_scores = F.softmax(scores - (1-mask)*1e20, dim=1)
                col_sa = pad.mul(normalized_scores.unsqueeze(2).expand_as(pad)).sum(1)
                table_col_reps.append(col_sa)
            table_col_reps = torch.cat(table_col_reps, dim=0)
            col_reps.append(table_col_reps)
            col_mask.append(torch.ones(len(table_col_reps)))

            # compute selfattn for this talbe
            scores = self.utt_table_sa_scorer(bert_table).squeeze(2)
            normalized_scores = F.softmax(scores - (1-mask_table)*1e20, dim=1)
            tab_sa = bert_table.mul(normalized_scores.unsqueeze(2).expand_as(bert_table)).sum(1)
            table_reps.append(tab_sa)
            table_mask.append(torch.ones(len(tab_sa)))

        col_reps = nn.utils.rnn.pad_sequence(col_reps, batch_first=True, padding_value=0)
        col_mask = nn.utils.rnn.pad_sequence(col_mask, batch_first=True, padding_value=0).to(self.device)
        table_reps = nn.utils.rnn.pad_sequence(table_reps, batch_first=True, padding_value=0)
        table_mask = nn.utils.rnn.pad_sequence(table_mask, batch_first=True, padding_value=0).to(self.device)

        col_trans, _ = rnn.run_rnn(self.utt_col_trans, col_reps, col_mask.sum(1).long())
        table_trans, _ = rnn.run_rnn(self.utt_table_trans, table_reps, table_mask.sum(1).long())
        table_trans = self.dropout(table_trans)

        cand = self.dropout(torch.cat([
            self.utt_emb.weight.unsqueeze(0).repeat(B, 1, 1),
            col_trans,
        ], dim=1))
        cand_mask = torch.cat([
            torch.ones(B, len(self.utt_vocab)).float().to(self.device),
            col_mask
        ], dim=1)

        utt_dec = self.utt_pointer_decoder(
            emb=cand, emb_mask=cand_mask,
            enc=table_trans, enc_mask=table_mask.float(),
            state0=None,
            gt=utt_pointer if self.training else None,
            max_len=self.args.max_query_len,
            batch=batch,
        )
        return dict(query_dec=query_dec, value_dec=value_dec, utt_dec=utt_dec)

    def extract_preds(self, out, feat, batch):
        preds = {}
        for pointer, value_pointer, utt_pointer, ex in zip(out['query_dec'].max(2)[1].tolist(), out['value_dec'].max(2)[1].tolist(), out['utt_dec'].max(2)[1].tolist(), batch):
            toks, value_toks = preprocess.SQLDataset.recover_query(pointer, ex['cands_query'], value_pointer, ex['cands_value'], voc=self.sql_vocab)
            post = toks[:]
            values = '___'.join(value_toks).split('SEP')
            values = [self.bert_tokenizer.convert_tokens_to_string(t.split('___')) for t in values if t]
            schema = self.database_schemas[ex['db_id']]
            try:
                post = editsql_postprocess.postprocess_one(' '.join(post), schema)
                if self.args.keep_values:
                    post = self.postprocess_value(post, values)
            except Exception as e:
                post = repr(e)

            utt_toks = preprocess.SQLDataset.recover_question(utt_pointer, ex['cands_question'], voc=self.utt_vocab)
            utt = self.bert_tokenizer.convert_tokens_to_string(utt_toks).replace(' id', '')
            preds[ex['id']] = dict(
                query_pointer=pointer,
                query_toks=toks,
                query=post,
                value_pointer=value_pointer,
                value_toks=value_toks,
                values=values,
                utt_pointer=utt_pointer,
                utt_toks=utt_toks,
                utt=utt,
            )
        return preds

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

    def compute_official_eval(self, dev, dev_preds):
        metrics = dict(official_em=0, official_ex=0)
        for ex in tqdm.tqdm(dev, desc='official eval'):
            p = dev_preds[ex['id']]
            g_str = ex['g_query']
            db_name = ex['db_id']
            db = os.path.join('data', 'database', db_name, db_name + ".sqlite")
            p_str = dev_preds[ex['id']]['query']
            # fix spacing
            spacing = [
                ('` ` ', '"'), ("''", '"'),
                ('> =', '>='), ('< =', '<='),
                ("'% ", "'%"), (" %'", "%'"),
            ]
            for f, t in spacing:
                p_str = p_str.replace(f, t)
            # recover casing
            for v in ex['g_values']:
                v = self.bert_tokenizer.convert_tokens_to_string(v).strip(' ' + string.punctuation)
                p_str = p_str.replace(v.lower(), v)
            schema = evaluation.Schema(evaluation.get_schema(db))

            p_sql = self.build_sql(schema, p_str, self.kmaps[ex['db_id']])

            # the offical eval script is buggy and modifies arguments in place
            try:
                em = self.evaluator.eval_exact_match(copy.deepcopy(p_sql), copy.deepcopy(ex['g_sql']))
            except Exception as e:
                em = False
            # if not em:
            #     print(g_str)
            #     print(p_str)
            #     print(ex['final_sql_parse'])
            #     import pdb; pdb.set_trace()
            metrics['official_em'] += em

            if self.args.keep_values:
                g_ex = self.execute(db, g_str, ex['g_sql'])
                p_ex = self.execute(db, p_str, p_sql)
                exe = 0 if p_ex is False else p_ex == g_ex
                metrics['official_ex'] += exe

        metrics['official_em'] /= len(dev)
        metrics['official_ex'] /= len(dev)
        return metrics

    def compute_metrics(self, data, preds):
        em = 0
        hyps, refs = [], []
        utt_hyps, utt_refs = [], []
        for ex in data:
            p = preds[ex['id']]
            gsql = ex['g_query_recov'].lower().split()
            psql = p['query'].lower().split()
            refs.append([gsql])
            hyps.append(psql)
            em += psql == gsql

            utt_hyps.append(p['utt_toks'])
            utt_refs.append([ex['g_question_toks']])
        metrics = {
            'em': em/len(data),
            'bleu': corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method3),
            'utt_bleu': corpus_bleu(utt_refs, utt_hyps, smoothing_function=SmoothingFunction().method3),
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

        dec = out['utt_dec']
        gold = feat['utt_pointer']
        utt = F.cross_entropy(dec.view(gold.numel(), -1), gold.view(-1), ignore_index=self.pad_id)
        losses['utt'] = utt / 2
        return losses

    def get_debug(self, ex, pred):
        gold_query = ex['g_query'].split()
        pred_query = pred['query_toks']
        pred_values = pred['value_toks']
        return dict(
            query=pred_query,
            gold_query=gold_query,
            match=pred_query == gold_query,
            values=pred_values,
            utt=pred['utt'],
            gold_utt=ex['g_question_toks'],
        )

    def postprocess_value(self, post, values):
        i = 0
        while i < len(values):
            v = values[i]
            m = re.search(r'\s+1', post)
            if m is None:
                break
            post = post[:m.start()] +  ' ' + v.replace("' ", "'").replace(" '", "'") + post[m.end():]
            i += 1
        return post

    def compute_upperbound(self, dev):
        preds = {}
        for ex in dev:
            toks, value_toks = preprocess.SQLDataset.recover_query(ex['pointer_query'], ex['cands_query'], ex['pointer_value'], ex['cands_value'], voc=self.sql_vocab)
            post = editsql_postprocess.postprocess_one(' '.join(toks), self.database_schemas[ex['db_id']])
            values = '___'.join(value_toks).split('SEP')
            values = [self.bert_tokenizer.convert_tokens_to_string(t.split('___')) for t in values if t]

            # apply fix
            for i, v in enumerate(values):
                words = v.split()
                fixed = [preprocess.value_replace.get(w, w) for w in words]
                values[i] = ' '.join(fixed)

            if self.args.keep_values:
                post = self.postprocess_value(post, values)

            utt_toks = preprocess.SQLDataset.recover_question(ex['pointer_question'], ex['cands_question'], voc=self.utt_vocab)
            utt = ' '.join(utt_toks)

            preds[ex['id']] = dict(
                query_pointer=ex['pointer_query'],
                query_toks=toks,
                query=post,
                utt_toks=utt_toks,
                utt=utt,
                value_toks=value_toks,
                values=values,
            )
        self.eval()
        metrics = self.compute_metrics(dev, preds)
        return metrics

    @classmethod
    def prune_train(cls, train, args, limit=800):
        orig = len(train)
        train = train.keep(lambda ex: sum(len(t['toks']) for t in ex['query_context']) < limit)
        train = train.keep(lambda ex: sum(len(t['toks']) for t in ex['question_context']) < limit)
        print('pruned from {} to {}'.format(orig, len(train)))
        return train

    @classmethod
    def prune_dev(cls, dev, args):
        return cls.prune_train(dev, args, limit=1000)

    @classmethod
    def get_stats(cls, splits, ext):
        stats = {}
        all_data = []
        for k, v in splits.items():
            stats[k] = len(v)
            all_data.extend(v)
        query_context_len = [sum(len(t['toks']) for t in ex['query_context']) for ex in all_data]
        question_context_len = [sum(len(t['toks']) for t in ex['question_context']) for ex in all_data]
        question_inp_len = [len(ex['value_context']) for ex in all_data]
        query_len = [len(ex['pointer_query']) for ex in splits['train']]
        value_len = [len(ex['pointer_value']) for ex in splits['train']]
        question_out_len = [len(ex['pointer_question']) for ex in splits['train']]
        stats.update({
            'sql': ext['sql_voc'],
            'utt': ext['utt_voc'],
            'max_query_context': max(query_context_len),
            'max_question_context': max(question_context_len),
            'max_question_inp': max(question_inp_len),
            'max_question_out': max(question_out_len),
            'max_query_len': max(query_len),
            'max_value_len': max(value_len),
        })
        return stats
