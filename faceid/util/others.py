import os
import csv
import sys
import shutil
import random
import logging
import numpy as np
from os.path import join, getsize, islink
from easydict import EasyDict as Edict
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.backends import cudnn

from . import dir_tool

# =========#=========#=========#=========#=========#=========#=========#=========
__all__ = ['set_logger', 'init_random', 'set_deterministic', 'AverageMeter']


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_loss_vs_lr(out_dir, losses, log_lrs):
    with open(join(out_dir, 'loss_vs_lr.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(losses)
        writer.writerow(log_lrs)
        csvfile.close()

    plt.figure()
    plt.plot(log_lrs, losses, label='ce_loss')
    plt.savefig(join(out_dir, 'loss_vs_lr.jpg'))

def init_random(seed=42, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

def set_deterministic(cudnn_benchmark):
    if cudnn_benchmark:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
                '%(asctime)s:%(levelname)s:%(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def get_size(din):
    total_size = 0
    for root, dirnames, filenames in os.walk(din):
        for fname in filenames:
            fpath = join(root, fname)
            if not islink(fpath):
                total_size += getsize(fpath)

    return total_size

def del_empty_job(workspace):
    jobs = dir_tool.sub_dpaths(workspace)
    if jobs:
        for job in jobs:
            has_ckpt = any([f.endswith('.pth.tar') or f.endswith('.pth')
                            for f in dir_tool.sub_fnames(job)])
            if not has_ckpt and get_size(job) < 5e6:
                shutil.rmtree(job)

def prorate_lr(lrs: list, batch_size: int, base: int = 512):
    new_lrs = []
    for lr in lrs:
        new_lrs.append(float(lr * batch_size / float(base)))

    return new_lrs