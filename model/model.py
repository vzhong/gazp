import torch
import shutil
import time
import os
import json
import copy
import tqdm
import utils
import pprint
import random
from torch import nn
from collections import defaultdict
from loguru import logger
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


class Module(nn.Module):

    def __init__(self, args, ext=None):
        super().__init__()
        self.args = args
        self.ext = ext
        self.state = {}

    @property
    def device(self):
        if self.args.gpu:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def load_emb(self, mat, embedder='emb', force_load=False):
        assert hasattr(self, embedder), 'Network has no attribute "{}"'.format(embedder)
        emb = self.emb
        if force_load:
            emb.weight.data = mat.to(emb.weight.device).data
        else:
            emb.weight.data.copy_(mat.to(emb.weight.device).data)

    def place_on_device(self):
        if self.args.gpu:
            return self.to(torch.device('cuda'))
        return self

    @property
    def dout(self):
        return os.path.join(self.args.dexp, self.args.model, self.args.name)

    def save_config(self):
        with open(os.path.join(self.dout, 'config.json'), 'wt') as f:
            json.dump(vars(self.args), f, indent=2)

    def compute_loss(self, out, feat, batch):
        raise NotImplementedError()

    def get_optimizer(self, train):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        total_step = len(train) / self.args.batch * self.args.epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (total_step - step) / total_step)
        return optimizer, scheduler

        raise NotImplementedError()

    def extract_preds(self, out, feat, batch):
        raise NotImplementedError()

    def compute_metrics(self, data, preds):
        raise NotImplementedError()

    @classmethod
    def get_dataset(cls, args):
        raise NotImplementedError()

    def get_debug(self, ex, pred):
        return dict(pred=pred)

    def write_preds(self, dev, dev_preds, fout):
        with open(fout, 'wt') as f:
            debug = []
            for ex in dev:
                debug.append(self.get_debug(ex, dev_preds[ex['id']]))
            json.dump(debug, f, indent=2)

    def get_file(self, *args):
        return os.path.join(self.dout, *args)

    def run_train(self, train, dev, args=None, verbose=True, eval_train=False):
        args = args or self.args
        if not os.path.isdir(self.dout):
            os.makedirs(self.dout)
        logger_loop = logger.bind(type='train_loop')

        logger_loop.info('Using config\n{}'.format(args))
        self.save_config()

        flog = self.get_file('train.log')
        fmetrics = open(self.get_file('train.metrics.jsonl'), 'wt')
        fbestmetrics = self.get_file('train.best.json')

        logger_loop.info('Logging to {}'.format(flog))
        logger.add(flog, rotation='5 MB', mode='wt', level='INFO', format="{time} {extra[type]} -- {message}")

        optimizer, scheduler = self.get_optimizer(train)
        iteration = 0
        start_epoch = 0
        if args.resume:
            metrics = self.load_save(fname=args.resume, optimizer=None if args.restart_optim else optimizer, scheduler=None if args.restart_optim or args.restart_scheduler else scheduler)
            start_epoch = metrics['epoch']
            iteration = metrics['iteration']
            logger_loop.info('Resuming from {}'.format(args.resume))
            logger_loop.info(pprint.pformat(metrics))
            self.args = args

        train_preds = None
        best = {}
        fbest = os.path.join(self.dout, 'best.tar')

        for epoch in tqdm.trange(start_epoch, args.epoch, desc='epoch'):
            logger_loop.info('Starting train epoch {}'.format(epoch))
            loss = defaultdict(lambda: 0)

            self.eval()
            train = train.reset()

            self.train()
            timing = defaultdict(list)
            for batch in train.batch(args.batch, shuffle=True, verbose=verbose, desc='train'):
                start_time = time.time()

                self.state.update({'iteration': iteration, 'epoch': epoch})
                feat = self.featurize(batch)
                out = self.forward(**feat)

                forward_time = time.time()

                loss_ = self.compute_loss(out, feat, batch)
                loss_backprop = 0
                if isinstance(loss_, dict):
                    for k, v in loss_.items():
                        loss['loss_' + k] += v.item() * len(batch)
                        loss_backprop += v
                else:
                    loss['loss'] += loss_.item() * len(batch)
                    loss_backprop += loss_

                iteration += len(batch)
                loss_backprop.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

                old_params = copy.deepcopy(list(self.named_parameters()))
                optimizer.step()

                # running avg
                new_params = dict(self.named_parameters())
                interp = self.args.running_avg
                for name, op in old_params:
                    new_params[name].data.copy_(interp*op.data + (1-interp)*new_params[name].data)
                self.load_state_dict(new_params)

                scheduler.step()
                optimizer.zero_grad()

                backward_time = time.time()

                if not eval_train:
                    train_preds = train.accumulate_preds(train_preds, self.extract_preds(out, feat, batch))

                timing['forward'].append(forward_time - start_time)
                timing['backward'].append(backward_time - forward_time)
                timing['batch'].append(backward_time - start_time)
            if eval_train:
                train_preds = self.run_pred(train, args, verbose=verbose, desc='train_pred')
            metrics = {'epoch': epoch, 'iteration': iteration}
            for k, v in loss.items():
                metrics[k] = v / len(train)
            for k, v in timing.items():
                metrics['time_{}'.format(k)] = sum(v) / len(v)
            metrics.update({'train_{}'.format(k): v for k, v in self.compute_metrics(train, train_preds).items()})

            dev_preds = self.run_pred(dev, args, verbose=verbose, desc='dev_pred')
            metrics.update({'dev_{}'.format(k): v for k, v in self.compute_metrics(dev, dev_preds).items()})

            fmetrics.write(json.dumps(metrics) + '\n')
            logger_loop.info('\n' + pprint.pformat(metrics))

            if self.better(metrics, best):
                best.update(metrics)
                logger_loop.info('Found new best! Saving checkpoint')
                self.save(metrics, optimizer, scheduler, fbest)
                self.write_preds(dev, dev_preds, self.get_file('dev.preds.json'))
                with open(fbestmetrics, 'wt') as f:
                    json.dump(metrics, f, indent=2)

        logger_loop.info('Loading best checkpoint from {}'.format(fbest))
        metrics = self.load_save(fname=fbest)
        logger_loop.info(pprint.pformat(metrics))
        fmetrics.close()

    def better(self, metrics, best):
        raise NotImplementedError()

    def save(self, metrics, optimizer, scheduler, fname='best.tar'):
        for i in reversed(list(range(4))):
            fthis = fname.replace('.tar', '.{}.tar'.format(i))
            fnext = fname.replace('.tar', '.{}.tar'.format(i+1))
            if os.path.isfile(fthis):
                if os.path.isfile(fnext):
                    os.remove(fnext)
                print('moving param {} to {}'.format(fthis, fnext))
                shutil.copy(fthis, fnext)
        print('saving param to {}'.format(fthis))
        torch.save(self.state_dict(), fthis)
        print('saving to {}'.format(fname))
        torch.save({
            'args': self.args,
            'ext': self.ext,
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': metrics,
        }, fname)

    def load_save(self, optimizer=None, scheduler=None, fname='best.pt'):
        save = torch.load(fname, map_location='cpu')
        self.load_state_dict(save['model'])
        if optimizer is not None:
            optimizer.load_state_dict(save['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(save['scheduler'])
        return save['metrics']

    def average_saves(self, saves):
        params = defaultdict(list)
        for fname in saves:
            save = torch.load(fname, map_location='cpu')
            self.load_state_dict(save)
            for name, p in self.named_parameters():
                params[name].append(p)
        for name, ps in params.items():
            params[name] = sum(ps) / len(ps)
        self.load_state_dict(params)

    @classmethod
    def load_inst(cls, fresume, overwrite=None):
        binary = torch.load(fresume, map_location=torch.device('cpu'))
        args = binary['args']
        ext = binary['ext']
        for k, v in (overwrite or {}).items():
            setattr(args, k, v)
        Model = utils.load_module(args.model)
        if args.model == 'nl2sql':
            Reranker = utils.load_module(args.beam_rank)
            ext['reranker'] = Reranker(args, ext)
        m = Model(args, ext).place_on_device()
        m.load_save(fname=fresume)
        return m

    def run_pred(self, dev, args=None, verbose=True, desc='pred'):
        args = args or self.args
        self.eval()
        preds = None
        with torch.no_grad():
            for batch in dev.batch(args.batch, shuffle=False, verbose=verbose, desc=desc):
                feat = self.featurize(batch)
                out = self.forward(**feat)
                preds = dev.accumulate_preds(preds, self.extract_preds(out, feat, batch))
        return preds

    def run_interactive_pred(self, dev, args=None, verbose=True, desc='pred'):
        args = args or self.args
        self.eval()
        preds = None
        prev_query = defaultdict(lambda: ['none'])
        if args.dataset == 'sparc':
            import preprocess_nl2sql_sparc as preprocess
        elif args.dataset == 'cosql':
            import preprocess_nl2sql_cosql as preprocess
        else:
            raise NotImplementedError()

        with torch.no_grad():
            for ex in tqdm.tqdm(dev, desc=desc):
                ex_id, turn_id = ex['id'].split('/')[1].split(':')
                prev = prev_query[ex_id]
                query_context = preprocess.SQLDataset.build_contexts(ex['g_question_toks'], prev, self.conv.database_schemas[ex['db_id']], self.bert_tokenizer)
                cands_query = preprocess.SQLDataset.make_column_cands(query_context)
                new = ex.copy()
                new['query_context'] = query_context
                new['cands_query'] = cands_query
                new['cands_query'], new['cands_value'] = preprocess.SQLDataset.make_cands(new, self.sql_vocab)
                batch = [new]
                feat = self.featurize(batch)
                out = self.forward(**feat)
                p = self.extract_preds(out, feat, batch)
                prev_query[ex_id] = p[new['id']]['query_toks']
                preds = dev.accumulate_preds(preds, p)
        return preds

    @classmethod
    def get_default_parser(cls, lr, batch, epoch, model, max_grad_norm=20, n_proc=1, seed=0):
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('--model', default=model, help='model to use')
        parser.add_argument('--lr', default=lr, help='learning rate', type=float)
        parser.add_argument('--max_grad_norm', default=max_grad_norm, help='grad norm clipping', type=float)
        parser.add_argument('--batch', default=batch, help='batch size', type=int)
        parser.add_argument('--epoch', default=epoch, help='epoch', type=int)
        parser.add_argument('--name', '-n', help='name for the experiment', default='default')
        parser.add_argument('--gpu', '-g', help='use GPU', action='store_true')
        parser.add_argument('--dexp', help='where to store the experiment', default='exp')
        parser.add_argument('--resume', help='checkpoint to resume from')
        parser.add_argument('--debug', help='run in debug mode', action='store_true')
        parser.add_argument('--seed', help='random seed', default=seed, type=int)
        parser.add_argument('--n_proc', help='how many processes to use for preprocessing', default=n_proc, type=int)
        parser.add_argument('--restart_optim', action='store_true', help='do not resume optimizer states')
        parser.add_argument('--restart_scheduler', action='store_true', help='do not resume scheduler states')
        return parser
