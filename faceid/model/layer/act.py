import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class FTSwishPlus(nn.Module):
    """
    Source   : https://github.com/lessw2020/FTSwishPlus
    """
    def __init__(self, threshold=-0.25, mean_shift=-0.1):
        super(FTSwishPlus,self).__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x):
        x = F.relu(x) * torch.sigmoid(x) + self.threshold
        if self.mean_shift is not None:
            x.sub_(self.mean_shift)
        return x


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                    stride=1, padding=1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        return torch.max(x, x1)


class Hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


def create_act(act_func, **kwargs):
    if act_func == 'prelu':
        return nn.PReLU()
    elif act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'mish':
        return Mish()
    elif act_func == 'hswish':
        return Hswish()
    elif act_func == 'ftswish':
        return FTSwishPlus()
    elif act_func == 'frelu':
        in_channel = kwargs.get('in_channel')
        return FReLU(in_channel)
