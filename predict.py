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
parser.add_argument('--resumes', nargs='*')
parser.add_argument('--dataset', default='spider', choices=['spider', 'cosql', 'sparc'])
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

    if orig_args.resumes:
        m.average_saves(orig_args.resumes)
    else:
        m.load_save(fname=fresume)

    # preprocess data
    data = dataset.Dataset()

    if orig_args.dataset == 'spider':
        import preprocess_nl2sql as preprocess
    elif orig_args.dataset == 'sparc':
        import preprocess_nl2sql_sparc as preprocess
    elif orig_args.dataset == 'cosql':
        import preprocess_nl2sql_cosql as preprocess

    proc_errors = set()
    with open(orig_args.input) as f:
        C = preprocess.SQLDataset
        raw = json.load(f)
        # make contexts and populate vocab
        for i, ex in enumerate(raw):
            for k in ['query', 'query_toks', 'query_toks_no_value', 'sql']:
                if k in ex:
                    del ex[k]
            ex['id'] = '{}/{}'.format(ex['db_id'], i)
            new = C.make_example(ex, m.bert_tokenizer, m.sql_vocab, m.conv.kmaps, m.conv, train=False, evaluation=True)
            new['question'] = ex['question']
            if new is not None:
                new['cands_query'], new['cands_value'] = C.make_cands(new, m.sql_vocab)
                data.append(new)
            else:
                print('proc error')
                proc_errors.add(ex['id'])

    # run preds
    if orig_args.dataset in {'cosql', 'sparc'}:
        preds = m.run_interactive_pred(data, args, verbose=True)
        raise NotImplementedError()
    else:
        preds = m.run_pred(data, args, verbose=True)
        assert len(preds) + len(proc_errors) == len(data), 'got {} predictions for {} examples'.format(len(preds), len(data))
        #  print('writing to {}'.format(orig_args.output))
        with open(orig_args.output, 'wt') as f:
            for ex in data:
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
