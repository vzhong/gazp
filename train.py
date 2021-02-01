import os
import json
import torch
import pprint
import utils
import dataset
from model.model import Module


mydir = os.path.dirname(__file__)


def get_parser():
    parser = Module.get_default_parser(lr=5e-5, batch=10, epoch=50, model='nl2sql', seed=3)
    parser.add_argument('--dataset', default='spider', choices=('spider', 'sparc', 'cosql'), help='dataset to use')
    parser.add_argument('--warmup', default=0.1, help='bert warm up rate', type=float)
    parser.add_argument('--bert_dropout', default=0.3, help='dropout rate', type=float)
    parser.add_argument('--dropout', default=0.5, help='dropout rate', type=float)
    parser.add_argument('--dec_dropout', default=0.2, help='dropout rate', type=float)
    parser.add_argument('--drnn', default=600, help='decoder rnn size', type=int)
    parser.add_argument('--demb', default=400, help='decoder emb size', type=int)
    parser.add_argument('--num_layers', default=2, help='decoder layers', type=int)
    parser.add_argument('--beam_size', default=0, help='beam search size', type=int)
    parser.add_argument('--beam_rank', default='rank_max', choices=('rank_max', 'rank_oracle', 'rank_pragmatic'), help='beam search ranker')
    parser.add_argument('--max_query_len', default=50, help='max query length for generation', type=int)
    parser.add_argument('--max_value_len', default=20, help='max value length for generation', type=int)
    parser.add_argument('--dcache', default=os.path.join(mydir, 'cache', 'bert'), help='cache directory')
    parser.add_argument('--fcache', default=None, help='cache data file')
    parser.add_argument('--test_only', action='store_true', help='only run test eval')
    parser.add_argument('--interactive_eval', action='store_true', help='evaluate using model-predicted previous query')
    parser.add_argument('--write_test_pred', help='file to write test preds to')
    parser.add_argument('--aug', nargs='*', help='augmentation data')
    parser.add_argument('--resumes', nargs='*', help='resume from many')
    parser.add_argument('--keep_values', action='store_true', help='do not strip out values')
    parser.add_argument('--skip_upperbound', action='store_true', help='do not compute upperbound')
    parser.add_argument('--aug_lim', type=int, help='how many aug examples to use')
    parser.add_argument('--fparser', default='exp/nl2sql/default/best.pt', help='parser model to use for nl generation')
    parser.add_argument('--tables', default='data/tables.json', help='tables dir')
    parser.add_argument('--db', default='data/database', help='SQLite database folder')
    parser.add_argument('--running_avg', default=0, type=float, help='portion of old parameters to mix in')
    parser.add_argument('--lambda_backward', default=0.8, help='weight for backward model', type=float)
    return parser


def main(args):
    args.gpu = torch.cuda.is_available()
    utils.manual_seed(args.seed)
    Model = utils.load_module(args.model)
    cache_file = args.fcache or (os.path.join('cache', 'data_{}_{}.debug.pt'.format(args.model, args.dataset) if args.debug else 'data_{}_{}.pt'.format(args.model, args.dataset)))
    splits, ext = torch.load(cache_file, map_location=torch.device('cpu'))
    splits = {k: dataset.Dataset(v) for k, v in splits.items()}
    splits['train'] = Model.prune_train(splits['train'], args)
    splits['dev'] = Model.prune_dev(splits['dev'], args)

    if args.model == 'nl2sql':
        Reranker = utils.load_module(args.beam_rank)
        ext['reranker'] = Reranker(args, ext)
    m = Model(args, ext).place_on_device()

    d = m.get_file('')
    if not os.path.isdir(d):
        os.makedirs(d)

    pprint.pprint(m.get_stats(splits, ext))

    if not args.test_only:
        if not args.skip_upperbound:
            print('upperbound')
            pprint.pprint(m.compute_upperbound(splits['train'][:1000]))
        if args.aug:
            augs = []
            for a in args.aug:
                augs.extend(torch.load(a))
            aug = dataset.Dataset(augs)
            splits['aug'] = Model.prune_train(aug, args)[:args.aug_lim]
            print('aug upperbound')
            pprint.pprint(m.compute_upperbound(aug[:10]))
            # aug_args = copy.deepcopy(args)
            # if 'consistent' not in args.aug:
            #     aug_args.epoch = 10
            # aug_dev = dataset.Dataset(random.sample(splits['train'], 3000))
            # m.run_train(aug, aug_dev, args=aug_args)
        pprint.pprint(m.get_stats(splits, ext))
        m.run_train(dataset.Dataset(splits['train'] + splits.get('aug', [])), splits['dev'], args=args)

    if args.resume:
        m.load_save(fname=args.resume)
    elif args.resumes:
        m.average_saves(args.resumes)
    if args.interactive_eval:
        dev_preds = m.run_interactive_pred(splits['dev'], args, verbose=True)
    else:
        dev_preds = m.run_pred(splits['dev'], args, verbose=True)

    if args.write_test_pred:
        with open(args.write_test_pred, 'wt') as f:
            json.dump(dev_preds, f, indent=2)
        print('saved test preds to {}'.format(args.write_test_pred))

    pprint.pprint(m.compute_metrics(splits['dev'], dev_preds))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
