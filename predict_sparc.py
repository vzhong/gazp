import os
import copy
import json
import torch
import pprint
import utils
import random
import dataset
import argparse
from model.model import Module


mydir = os.path.dirname(__file__)


parser = argparse.ArgumentParser()
parser.add_argument('resume')
parser.add_argument('input')
parser.add_argument('--dataset', default='sparc', choices=['cosql', 'sparc'])
parser.add_argument('--tables', default='tables.json')
parser.add_argument('--db', default='database')
parser.add_argument('--dcache', default='cache')
parser.add_argument('--batch', type=int, default=6)
parser.add_argument('--output', default='output.txt')


def main(orig_args):
    # load pretrained model
    fresume = os.path.abspath(orig_args.resume)
    # print('resuming from {}'.format(fresume))
    assert os.path.isfile(fresume), '{} does not exist'.format(fresume)

    orig_args.input = os.path.abspath(orig_args.input)
    orig_args.tables = os.path.abspath(orig_args.tables)
    orig_args.db = os.path.abspath(orig_args.db)
    orig_args.dcache = os.path.abspath(orig_args.dcache)

    binary = torch.load(fresume, map_location=torch.device('cpu'))
    args = binary['args']
    ext = binary['ext']
    args.gpu = torch.cuda.is_available()
    args.tables = orig_args.tables
    args.db = orig_args.db
    args.dcache =  orig_args.dcache
    args.batch = orig_args.batch
    Model = utils.load_module(args.model)
    if args.model == 'nl2sql':
        Reranker = utils.load_module(args.beam_rank)
        ext['reranker'] = Reranker(args, ext)
    m = Model(args, ext).place_on_device()
    m.load_save(fname=fresume)

    # preprocess data
    data = dataset.Dataset()

    if args.dataset == 'sparc':
        import preprocess_nl2sql_sparc as preprocess
    elif args.dataset == 'cosql':
        import preprocess_nl2sql_cosql as preprocess
    else:
        raise NotImplementedError()

    proc_errors = set()
    with open(orig_args.input) as f:
        C = preprocess.SQLDataset
        raw = json.load(f)

        # make contexts and populate vocab
        for i, ex in enumerate(raw):
            for turn_i, turn in enumerate(ex['interaction']):
                turn['id'] = '{}/{}:{}'.format(ex['database_id'], i, turn_i)
                turn['db_id'] = ex['database_id']
                for k in ['query', 'query_toks', 'query_toks_no_value', 'sql']:
                    if k in turn:
                        del turn[k]
                turn['question'] = turn['utterance']
                turn['g_question_toks'] = C.tokenize_question(turn['utterance'].split(), m.bert_tokenizer)
                turn['value_context'] = [m.bert_tokenizer.cls_token] + turn['g_question_toks'] + [m.bert_tokenizer.sep_token]
                turn['turn_i'] = turn_i
                data.append(turn)

    # run preds
    preds = m.run_interactive_pred(data, args, verbose=True)
    assert len(preds) == len(data), 'got {} predictions for {} examples'.format(len(preds), len(data))
    #  print('writing to {}'.format(orig_args.output))
    with open(orig_args.output, 'wt') as f:
        for i, ex in enumerate(data):
            if i != 0 and ex['turn_i'] == 0:
                f.write('\n')
            if ex['id'] in proc_errors:
                s = 'ERROR'
            else:
                p = preds[ex['id']]
                s = p['query']
            f.write(s + '\n')
        f.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
