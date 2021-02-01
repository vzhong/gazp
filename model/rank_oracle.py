import re
import os
import copy
import editsql_postprocess
import preprocess_nl2sql as preprocess
from eval_scripts import evaluation
from model.rank_max import Module as Base


class Module(Base):

    def match(self, beam, ex):
        pointer = beam.inds
        toks = preprocess.SQLDataset.recover_slots(pointer, ex['cands_query'], eos=self.sql_vocab.word2index('EOS'))
        db_name = ex['db_id']
        db_path = os.path.join('data', 'database', db_name, db_name + ".sqlite")
        g_str = ex['g_query']
        p_str = beam.post
        try:
            schema = evaluation.Schema(evaluation.get_schema(db_path))
            p_sql = preprocess.SQLDataset.build_sql(schema, p_str, self.kmaps[db_name])
            g_sql = preprocess.SQLDataset.build_sql(schema, g_str, self.kmaps[db_name])
            # the offical eval script is buggy and modifies arguments in place
            em = self.evaluator.eval_exact_match(copy.deepcopy(p_sql), copy.deepcopy(g_sql))
        except Exception as e:
            em = False
        beam.query_toks = toks
        beam.em = em
        beam.query = p_str
        beam.g_query = g_str
        if beam.toks == ex['g_query_norm'].split():
            beam.em = True
        return beam

    def rerank(self, beams, ex):
        ps = [self.match(b, ex) for b in beams]
        sort = sorted(ps, key=lambda b: (float(b.em), b.beam_score), reverse=True)
        return sort[0]
