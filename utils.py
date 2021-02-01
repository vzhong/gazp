import torch
import signal
import random
import os
import logging
import requests
import numpy as np
from torch import nn
from importlib import import_module


def pad_sequence(inds, pad, device=None):
    out = nn.utils.rnn.pad_sequence(inds, batch_first=True, padding_value=pad)
    return out if device is None else out.to(device)


def load_module(name, root='model'):
    return import_module('{}.{}'.format(root, name)).Module


def manual_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def download(url, fname, skip_if_exists=True):
    if os.path.isfile(fname):
        if skip_if_exists:
            logging.warn('File {} already exists. Skipping download.'.format(fname))
            return
        else:
            logging.warn('Removing existing file {}'.format(fname))
            os.remove(fname)
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return fname


def is_numerical_column(cells):
    for c in cells:
        try:
            float(cells)
        except:
            return False
    return True


all_loggers = {}


def get_logger(name, fout=None):
    global all_loggers
    if name not in all_loggers:
        all_loggers[name] = logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if fout is not None:
            fh = logging.FileHandler(fout)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger = all_loggers[name]
    return logger


class File:
    """
    Utilities for file IO
    """

    default_root = os.path.join(os.environ.get('HOME'), '.vml')
    ROOT = os.environ.get('VML_ROOT', default_root)

    @classmethod
    def new(cls, fname, ensure_dir=False):
        real = os.path.join(cls.ROOT, fname)
        parent = os.path.dirname(real)
        if ensure_dir and not os.path.isdir(parent):
            os.makedirs(parent)
        return real

    @classmethod
    def new_dir(cls, dirname, ensure_dir=False):
        dirname = os.path.join(cls.ROOT, dirname)
        if ensure_dir and not os.path.isdir(dirname):
            os.makedirs(dirname)
        return dirname


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
