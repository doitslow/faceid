from torch import nn
from .act import create_act

__all__ = ['Stem']


class Stem(nn.Module):
    def __init__(self, inplanes, bn_eps, bn_mom, act_func, net_in, **kwargs):
        super(Stem, self).__init__()
        self._net_in = net_in
        if self._net_in == '1':
            self.conv0 = nn.Conv2d(3, inplanes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn0 = nn.BatchNorm2d(inplanes, eps=bn_eps, momentum=bn_mom)
            act_kwargs = {}
            if act_func == 'frelu':
                act_kwargs['in_channel'] = inplanes
            self.act0 = create_act(act_func, **act_kwargs)

    def forward(self, x):
        if self._net_in == '1':
            x = self.conv0(x)
            x = self.bn0(x)
            x = self.act0(x)
        return x
