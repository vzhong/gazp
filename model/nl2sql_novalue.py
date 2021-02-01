import os
import re
import tqdm
import copy
import json
import utils
import torch
import string
import random
import sqlite3
import subprocess
import editsql_postprocess
import numpy as np
import embeddings as E
import preprocess_nl2sql_novalue as preprocess
from eval_scripts import evaluation
from vocab import Vocab
from torch import nn
from torch.nn import functional as F
from model import attention, rnn, decoder
from model.model import Module as Base
from bleu import corpus_bleu, SmoothingFunction
from collections import defaultdict, Counter
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, WarmupLinearSchedule


class Module(Base):

    def __init__(self, args, ext):
        super().__init__(args, ext)
        self.database_schemas = ext['database_schemas']
        self.database_content = ext['db_content']
        self.kmaps = ext['kmaps']
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(preprocess.BERT_MODEL, cache_dir=args.dcache)
        self.bert_embedder = DistilBertModel.from_pretrained(preprocess.BERT_MODEL, cache_dir=args.dcache)
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

        feat['utterance_mask'] = utils.pad_sequence([torch.ones(len(t)) for t in feat['utterance']], 0, self.device)
        feat['utterance'] = utils.pad_sequence(feat['utterance'], self.pad_id, self.device)

        feat['batch'] = batch
        return feat

    def forward(self, utterance, utterance_mask, tables, tables_mask, starts, ends, query_pointer, batch):
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

        return dict(query_dec=query_dec)

    def extract_preds(self, out, feat, batch):
        preds = {}
        for pointer, ex in zip(out['query_dec'].max(2)[1].tolist(), batch):
            toks = preprocess.SQLDataset.recover_query(pointer, ex['cands_query'], voc=self.sql_vocab)
            post = toks[:]
            schema = self.database_schemas[ex['db_id']]
            try:
                post = post_no_value = editsql_postprocess.postprocess_one(' '.join(post), schema)
            except Exception as e:
                post = post_no_value = repr(e)

            preds[ex['id']] = dict(
                query_pointer=pointer,
                query_toks=toks,
                query_no_value=post_no_value,
                query=post,
            )
        return preds

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
            schema = evaluation.Schema(evaluation.get_schema(db))
            p_sql = preprocess.SQLDataset.build_sql(schema, p_str, self.kmaps[ex['db_id']])

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
        metrics['official_em'] /= len(dev)
        metrics['official_ex'] /= len(dev)
        return metrics

    def compute_metrics(self, data, preds):
        em = 0
        hyps, refs = [], []
        for ex in data:
            p = preds[ex['id']]
            gsql = ex['g_query'].lower().split()
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
        return losses

    def get_debug(self, ex, pred):
        gold_query = ex['g_query']
        pred_query = pred['query']
        return dict(
            question=' '.join(ex['g_question_toks']),
            query=pred_query,
            gold_query=gold_query,
            match=pred_query == gold_query,
        )

    def compute_upperbound(self, dev):
        preds = {}
        for ex in dev:
            toks = preprocess.SQLDataset.recover_query(ex['pointer_query'], ex['cands_query'], voc=self.sql_vocab)
            post = editsql_postprocess.postprocess_one(' '.join(toks), self.database_schemas[ex['db_id']])

            preds[ex['id']] = dict(
                query_pointer=ex['pointer_query'],
                query_toks=toks,
                query=post,
            )
        self.eval()
        metrics = self.compute_metrics(dev, preds)
        return metrics

    @classmethod
    def prune_train(cls, train, args, limit=1000):
        orig = len(train)
        train = train.keep(lambda ex: sum(len(t['toks']) for t in ex['query_context']) < limit)
        print('pruned from {} to {}'.format(orig, len(train)))
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
        query_len = [len(ex['pointer_query']) for ex in splits['train']]
        stats.update({
            'sql': ext['sql_voc'],
            'max_query_context': max(query_context_len),
            'max_query_len': max(query_len),
        })
        return stats
