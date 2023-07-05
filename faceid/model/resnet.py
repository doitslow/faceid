import os

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import numpy as np

from .layer.norm import Ibn


debug_global = False
rank0 = int(os.environ['OMPI_COMM_WORLD_RANK']) == 0

def resnet_weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Act(nn.Module):
    def __init__(self, act_func):
        super(Act, self).__init__()
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.act_func = act_func

    def forward(self, x):
        if self.act_func == 'prelu':
            out = self.prelu(x)
        else:
            out = self.relu(x)
        return out


# =========#=========#=========#=========#=========#=========#=========#=========
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, bn_eps:float = 2e-5, bn_mom:float = 0.1,
                 act_func: str = "prelu", downsample=None, use_ibn=False):
        super(BasicBlock, self).__init__()
        self.use_ibn = use_ibn
        self.downsample = downsample
        self.planes = planes

        if self.use_ibn:
            self.ibn1 = Ibn(inplanes, bn_eps, bn_mom)
        else:
            self.bn1 = nn.BatchNorm2d(inplanes, eps=bn_eps, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_mom)
        self.act1 = Act(act_func)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_mom)

    def _forward_impl(self, x):
        if self.use_ibn:
            out = self.ibn1(x)
        else:
            out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        return out

    def forward(self, x):
        identity = x
        out = self._forward_impl(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out


class ResNet(nn.Module):
    def __init__(self, block, units, filters, img_size=112, embed_size=256,
                 dropout=0.4, act_func='prelu', use_ibn=True, **kwargs):
        super(ResNet, self).__init__()
        assert img_size % 16 == 0, "Size of last feature map not even!"
        self.bn_eps = kwargs.get('bn_eps', 2e-5)
        self.bn_mom = kwargs.get('bn_mom', 0.1)
        self.act_func = act_func
        self.embed_size = embed_size

        self.conv0 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(filters[0], eps=self.bn_eps, momentum=self.bn_mom)
        self.act0 = Act(act_func)

        self.layer1 = self._make_layer(block, filters[0], filters[1], units[0], use_ibn)
        self.layer2 = self._make_layer(block, filters[1], filters[2], units[1], use_ibn)
        self.layer3 = self._make_layer(block, filters[2], filters[3], units[2], use_ibn)
        self.layer4 = self._make_layer(block, filters[3], filters[4], units[3], False)

        # resnet for ImageNet uses average pooling to reduce feature map to 1*1
        self.bn1 = nn.BatchNorm2d(filters[4], eps=self.bn_eps, momentum=self.bn_mom)
        self.dropout = nn.Dropout(dropout)
        self.neck = nn.Linear(pow((int(img_size / 16)), 2) * filters[4], embed_size)
        self.bn2 = nn.BatchNorm1d(embed_size, eps=self.bn_eps, momentum=self.bn_mom, affine=False)

        self.apply(resnet_weight_init)

    def _make_layer(self, block, inplanes, planes, num_block, use_ibn=False):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes, eps=self.bn_eps, momentum=self.bn_mom),
        )
        layers = [block(inplanes, planes, 2, self.bn_eps, self.bn_mom, self.act_func,
                        downsample=downsample, use_ibn=use_ibn)]

        for i in range(1, num_block):
            layers.append(
                block(planes, planes, 1, self.bn_eps, self.bn_mom, self.act_func, use_ibn=use_ibn)
            )

        return nn.Sequential(*layers)

    def forward(self, x):         # input size is [N, 3, 112, 112]
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn1(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        x = self.neck(x.float())
        embed = self.bn2(x.float())

        return embed


def resnet50(**kwargs):

    return ResNet(BasicBlock, [3, 4, 14, 3], [64, 64, 128, 256, 512], **kwargs)
