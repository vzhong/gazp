import re
import os
import converter
import editsql_postprocess
import preprocess_nl2sql as preprocess
from eval_scripts import evaluation


class Module:

    def __init__(self, args, ext, remove_invalid=True):
        self.args = args
        self.remove_invalid = remove_invalid
        if self.remove_invalid:
            self.conv = converter.Converter(tables=getattr(args, 'tables', 'data/spider/tables'), db=getattr(args, 'db', 'data/database'))
            self.sql_vocab = ext['sql_voc']
            self.evaluator = evaluation.Evaluator()

    def parse_beam(self, beam, ex):
        pointer = beam.inds
        toks = beam.query_toks = preprocess.SQLDataset.recover_slots(pointer, ex['cands_query'], eos=self.sql_vocab.word2index('EOS'))
        post = toks[:]
        db_name = ex['db_id']
        db_path = os.path.join('data', 'database', db_name, db_name + ".sqlite")
        try:
            post = post_no_value = self.conv.recover(' '.join(post), db_name)
            post = post_no_value = re.sub('\s([0-9a-zA-Z_]+\.)\*\s', '', post)
        except Exception as e:
            beam.query = beam.sql = None
            return beam
        p_str = post

        # fix spacing
        spacing = [
            ('` ` ', '"'), ('` `', '"'), ("''", '"'),
            ('> =', '>='), ('< =', '<='), ('! =', '!='),
            ("'% ", "'%"), (" %'", "%'"),
        ]
        for f, t in spacing:
            post = post.replace(f, t)
        try:
            p_sql = self.conv.build_sql(p_str, db_name)
        except Exception as e:
            p_sql = None
        beam.query = p_str
        beam.sql = p_sql
        return beam

    def remove_invalid_beams(self, beams, ex):
        keep = []
        for b in beams:
            self.parse_beam(b, ex)
            if b.sql is not None:
                keep.append(b)
        return keep

    def rerank(self, beams, ex, key='beam_score'):
        valid = self.remove_invalid_beams(beams, ex) if self.remove_invalid else beams
        sort = sorted(valid, key=lambda b: getattr(b, key), reverse=True)
        return sort[0]
