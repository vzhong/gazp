import os
import pprint
import numpy as np
import json
from tqdm import tqdm
from utils import File


class Dataset(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __add__(self, rhs):
        return self.__class__(super().__add__(rhs))

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(result, list):
            return self.__class__(result)
        else:
            return result

    @classmethod
    def annotate(cls, tokenizer, din, dout, limit=None):
        raise NotImplementedError()

    @classmethod
    def download(cls, dout):
        raise NotImplementedError()

    @classmethod
    def pull(cls, tokenizer, limit=None):
        dout = File.new_dir(os.path.join('dataset', cls.__name__.lower()), ensure_dir=True)
        draw = os.path.join(dout, 'raw')
        dann = os.path.join(dout, 'ann')
        if not os.path.isdir(draw):
            os.makedirs(draw)
        cls.download(draw)
        if not os.path.isdir(dann):
            os.makedirs(dann)
        anns = cls.annotate(tokenizer, draw, dann, limit=limit)
        return anns

    def compute_metrics(self, preds):
        raise NotImplementedError()

    def accumulate_preds(self, preds, batch_preds):
        if preds is None:
            preds = batch_preds.copy()
        else:
            preds.update(batch_preds)
        return preds

    @classmethod
    def serialize_one(cls, ex):
        return json.dumps(ex)

    @classmethod
    def deserialize_one(cls, line):
        return json.loads(line)

    def save(self, fname, verbose=False):
        with open(fname, 'wt') as f:
            iterator = tqdm(self, desc='save') if verbose else self
            for ex in iterator:
                f.write(json.dumps(self.serialize_one(ex)) + '\n')

    @classmethod
    def load(cls, fname, limit=None):
        with open(fname) as f:
            data = [cls.deserialize_one(line) for i, line in enumerate(f) if limit is None or i < limit]
        return cls(data)

    def keep(self, keep):
        return self.__class__([e for e in self if keep(e)])

    def batch(self, batch_size, shuffle=False, verbose=False, desc='batch'):
        items = self[:]
        if shuffle:
            np.random.shuffle(items)
        iterator = range(0, len(items), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=desc)
        for i in iterator:
            yield items[i:i+batch_size]

    def reset(self):
        return self


class AugDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AugModel = self.train_data = self.generator = self.size = self.p_gen = None
        
    def set_augmenter(self, AugModel, generator):
        self.AugModel = AugModel
        self.generator = generator
        return self

    def set_data(self, size, p_gen, train_data):
        self.size = size
        self.p_gen = p_gen
        self.train_data = train_data
        return self

    def reset(self):
        print('Generating data')
        self.clear()
        gen = self.generator.run_gen(self.size, self.AugModel)
        self.extend(gen)
        print('Generated data size {}'.format(len(self)))
        return self

    def batch(self, batch_size, shuffle=False, verbose=False, desc='batch'):
        gen_batch_size = int(self.p_gen * batch_size)
        train_batch_size = batch_size - gen_batch_size
        for train in self.train_data.batch(train_batch_size, shuffle=shuffle, verbose=verbose, desc=desc):
            sample_inds = np.random.choice(list(range(len(self))), size=gen_batch_size)
            gen = train.__class__([self[i] for i in sample_inds])
            yield gen + train

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(result, list):
            new = self.__class__(result)
            new.set_augmenter(self.AugModel, self.generator).set_data(self.size, self.p_gen, self.train_data)
            return new
        else:
            return result
