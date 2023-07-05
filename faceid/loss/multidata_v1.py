import os
import logging
import os.path as op
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
from util.dist_ops import all_gather
# from regularize import isda


# wsize = int(os.environ['OMPI_COMM_WORLD_SIZE'])
# WRANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
# LRANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


# =========#=========#=========#=========#=========#=========#=========#=========
# LIUJIN: multiple weights for multiple training datasets
"""
    - backbone: stem + body + neck (optional: producing embedding)
    - head: everything from embedding to predictions
"""

class MultidataLoss(nn.Module):
    def __init__(self, name, embed_size, num_classes: list, **kwargs):
        super(MultidataLoss, self).__init__()
        logging.info("Initializing {} for multi data loss.".format(type(self)))
        self.num_classes = num_classes
        self.margin_s = kwargs.get('margin_s')

        self.multi_weights = nn.ParameterList(
            [Parameter(torch.FloatTensor(i, embed_size)) for i in num_classes])

        for i in range(len(num_classes)):
            std = math.sqrt(2.0 / float(num_classes[i] + embed_size))
            a = math.sqrt(3.0) * std
            nn.init._no_grad_uniform_(self.multi_weights[i], -a, a)

        if 'arcface' in name:
            self.loss = ArcMarginProduct(**kwargs)   # Arcface loss
        elif 'multi_margin' in name:
            self.loss = MultiMarginProduct(**kwargs)  # Combined loss
        elif 'cosface' in name:
            self.loss = CosProduct(**kwargs)

    def forward(self, embed, label, dataset_id=0, wsize=1, wrank=0, isda_params=None):
        cosine = F.linear(F.normalize(embed), F.normalize(self.multi_weights[dataset_id]), bias=None)
        cosine = cosine.float().clamp(-1, 1)    # for numerical stability
        scaled_cosine = cosine.detach().clone() * self.margin_s
        output = self.loss(cosine, label, self.num_classes[dataset_id])   # margin-based loss need label for selective margin enlargement

        return output


class MultidataPartialFC(nn.Module):
    def __init__(self, name, embed_size, num_classes, wsize, **kwargs):
        super(MultidataPartialFC, self).__init__()
        logging.info("Initializing {} for multi data loss.".format(type(self)))
        self.num_classes = num_classes
        self.margin_s = kwargs.get('margin_s')

        fc_lengths = [math.ceil(num / wsize) for num in num_classes]
        self.multi_weights = nn.ParameterList(
            [Parameter(torch.FloatTensor(i, embed_size)) for i in fc_lengths])

        for i in range(len(num_classes)):
            std = math.sqrt(2.0 / float(num_classes[i] + embed_size))
            a = math.sqrt(3.0) * std
            nn.init._no_grad_uniform_(self.multi_weights[i], -a, a)

        if 'arcface' in name:
            self.loss = ArcMarginProduct(**kwargs)   # Arcface loss
        elif 'multi_margin' in name:
            self.loss = MultiMarginProduct(**kwargs)  # Combined loss
        elif 'cosface' in name:
            self.loss = CosProduct(**kwargs)

    def forward(self, embed, label, dataset_id=0, wsize=1, wrank=0, isda_params=None):
        embd_all = torch.cat(all_gather(embed), dim=0)
        # calculate cosine for all samples across all GPUs with local weight
        cosine_local = F.linear(F.normalize(embd_all),
                                F.normalize(self.multi_weights[dataset_id]), bias=None)
        # gather cosines from all GPUs and chop each local cosine along batch size dimension
        cosine_gather_chunk = [torch.chunk(c, wsize, dim=0)[wrank]
                               for c in all_gather(cosine_local)]
        # concatenate all consine chunks along label dimension
        cosine = torch.cat(cosine_gather_chunk, dim=1)

        cosine = cosine.float().clamp(-1, 1)    # for numerical stability
        scaled_cosine = cosine.detach().clone() * self.margin_s
        output = self.loss(cosine, label)   # margin-based loss need label for selective margin enlargement

        return output


class CosProduct(nn.Module):
    def __init__(self, **kwargs):
        super(CosProduct, self).__init__()
        self.s = 64.0
        self.m = 0.35
        # label smoothing
        # assert 0 <= conf.margin_smoothing < 1
        self.smoothing = kwargs.get('margin_smoothing')
        logging.info("Initializing CosProduct for multi data loss.")

    def forward(self, cosine, label):
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=cosine.get_device())
        if self.smoothing == 0.0:
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        else:
            one_hot.fill_(self.smoothing / (cosine.size(1) -1))
            one_hot.scatter_(1, label.view(-1, 1).long(), 1 - self.smoothing)

        output = ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

        return output


class ArcMarginProduct(nn.Module):
    def __init__(self, **kwargs):
        super(ArcMarginProduct, self).__init__()
        self.s = kwargs.get('margin_s')
        self.m = kwargs.get('margin_m_arc')
        self.easy_margin = kwargs.get('easy_margin')
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        assert self.s > 0.0
        assert self.m >= 0.0
        assert self.m < (math.pi / 2)
        # label smoothing
        # assert 0 <= conf.margin_smoothing < 1
        self.smoothing = kwargs.get('margin_smoothing')

    def forward(self, cosine, label):
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=cosine.get_device())
        if self.smoothing == 0.0:
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        else:
            one_hot.fill_(self.smoothing / (cosine.size(1) - 1))
            one_hot.scatter_(1, label.view(-1, 1).long(), 1 - self.smoothing)

        output = ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

        return output


# =========#=========#=========#=========#=========#=========#=========#=========
class MultiMarginProduct(nn.Module):
    def __init__(self, **kwargs):
        super(MultiMarginProduct, self).__init__()
        self.s = kwargs.get('margin_s')
        self.m_arc = kwargs.get('margin_m_arc')
        self.m_cos = kwargs.get('margin_m_cos')
        self.m_sphere = kwargs.get('margin_m_sphere')
        self.easy_margin = kwargs.get('easy_margin')

        self.iter = 0
        self.lambda_min = 5.0
        self.base = 1000
        self.alpha = 0.0001
        self.power = 2
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        self.cos_m1 = math.cos(self.m_arc)
        self.sin_m1 = math.sin(self.m_arc)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m_arc)
        self.mm = math.sin(math.pi - self.m_arc) * self.m_arc

        # label smoothing
        # assert 0 <= conf.margin_smoothing < 1
        self.smoothing = kwargs.get('margin_smoothing')

    # =========#=========#=========#=========#=========#=========#=========#=========
    def forward(self, cosine, label, num_cls):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.alpha * self.iter)
                              ** (-1 * self.power))
        cosine_m = self.margin_formula[int(self.m_sphere)](cosine)  # different margin * gt
        theta = cosine.data.acos()  # inverse cosine function, get theta
        k = ((self.m_sphere * theta) / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cosine_m - 2 * k
        cosine = (self.cur_lambda * cosine + phi_theta) / (1 + self.cur_lambda)

        # cos(theta + m1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m1 - sine * self.sin_m1

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=cosine.get_device())
        if self.smoothing == 0.0:
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        else:
            one_hot.fill_(self.smoothing / (cosine.size(1) - 1))
            one_hot.scatter_(1, label.view(-1, 1).long(), 1 - self.smoothing)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output - one_hot * self.m_cos  # additive cosine margin
        output *= self.s

        return output
