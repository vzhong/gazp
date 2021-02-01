import re
from model.nl2sql import Module as Base, DistilBertModel, DistilBertTokenizer
import preprocess_nl2sql_sparc as preprocess


class Module(Base):

    def __init__(self, args, ext):
        super().__init__(args, ext)
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(preprocess.BERT_MODEL + '/vocab.txt', cache_dir=args.dcache)
        self.bert_embedder = DistilBertModel.from_pretrained(preprocess.BERT_MODEL, cache_dir=args.dcache)
        self.value_bert_embedder = DistilBertModel.from_pretrained(preprocess.BERT_MODEL, cache_dir=args.dcache)

    def extract_query(self, pointer, value_pointer, ex):
        toks, value_toks = preprocess.SQLDataset.recover_query(pointer, ex['cands_query'], value_pointer, ex['cands_value'], voc=self.sql_vocab)
        post = toks[:]
        values = self.recover_values(value_toks)

        q_low = ex['question'].lower()
        for i, v in enumerate(values):
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

            # apply fix
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

    def run_pred(self, dev, args=None, verbose=True, desc='pred', force_non_interactive=False):
        if 'train' in desc:
            force_non_interactive = True
        if force_non_interactive:
            return super().run_pred(dev, args=args, verbose=verbose, desc=desc)
        else:
            return super().run_interactive_pred(dev, args=args, verbose=verbose, desc=desc)
