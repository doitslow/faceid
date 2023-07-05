from torch import nn

__all__ = ['Head']


class Head(nn.Module):
    """
        Defualt head type is 'E'
    """
    def __init__(self, inplanes, emb_size, drop_rate, bn_eps, bn_mom, net_out,
                 **kwargs):
        super(Head, self).__init__()
        self.drop = drop_rate
        self.net_out = net_out
        if self.net_out == 'E':
            self.bn0 = nn.BatchNorm2d(inplanes, eps=bn_eps, momentum=bn_mom)
            if self.drop:
                self.dropout = nn.Dropout(drop_rate)
            self.fc0 = nn.Linear(inplanes * 7 * 7, emb_size)
            self.bn1 = nn.BatchNorm1d(emb_size, eps=bn_eps, momentum=bn_mom,
                                      affine=False)

    def forward(self, x):
        if self.net_out == 'E':
            x = self.bn0(x)
            x = x.view(x.size(0), -1)
            if self.drop:
                x = self.dropout(x)
            x = self.fc0(x.float())
            x = self.bn1(x.float())
        return x

