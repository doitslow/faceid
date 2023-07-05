import os
import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['DropBlock2D']


class DropBlock2D(nn.Module):
    """
        Dropblock rate is increased after every epoch ==> from ResNeSt implementation
    """
    def __init__(self, db_rate, block_size, steps, factors):
        super(DropBlock2D, self).__init__()
        self.db_size = block_size
        self.steps = steps
        self.values = factors * db_rate
        self.db_prob = 0.   # drop rate always start with 0.
        self.step = 0

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
        if self.step in self.steps:
            # make sure self.db_prob is a number NOT numpy array
            self.db_prob = self.values[np.where(self.steps == self.step)].item()
            if int(os.environ['OMPI_COMM_WORLD_RANK']) == 0:
                logging.info("droplock drop rate changed to {}".format(self.db_prob))
        self.step += 1

        if not self.training or self.db_prob == 0.:
            return x
        else:
            b, c, h, w = x.shape
            coeff = h ** 2 / ((self.db_size ** 2) * (h - self.db_size + 1) ** 2)
            gamma = self.db_prob * coeff
            p = torch.ones(1, c, h, w).to(x.device) * gamma
            mask = 1 - F.max_pool2d(torch.bernoulli(p),
                                    kernel_size=(self.db_size, self.db_size),
                                    stride=(1, 1),
                                    padding=(self.db_size //2, self.db_size //2))
            return mask * x * (mask.numel() / (mask.sum() + 1e-8))