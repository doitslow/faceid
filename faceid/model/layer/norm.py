import torch
import torch.nn as nn
import torch.nn.functional as F

from .act import Hsigmoid

__all__ = ['Ibn', 'AttenNorm', 'MixtureBatchNorm2d']


class Ibn(nn.Module):
    def __init__(self, inplanes, eps, bn_mom):
        super(Ibn, self).__init__()
        self.in1 = nn.InstanceNorm2d(int(inplanes/2), eps=eps)
        self.bn1 = nn.BatchNorm2d(int(inplanes/2), eps=eps, momentum=bn_mom)

    def forward(self, x):
        split = torch.split(x, int(x.size()[1]/2), dim=1)
        out1 = self.in1(split[0])
        out2 = self.bn1(split[1])
        out = torch.cat([out1, out2], 1)
        return out


class AttenNorm(nn.Module):
    def __init__(self, inplanes, nClass=16, kama=10, orth_lambda=1e-3, eps=1e-7):
        super(AttenNorm, self).__init__()
        self.nClass = nClass
        self.kama = kama
        self.orth_lambda = orth_lambda
        self.eps = eps
        self.inplanes = inplanes

        self.sn_key_conv = nn.Conv2d(inplanes, inplanes//8, 1, 1)
        self.sn_query_conv = nn.Conv2d(inplanes, inplanes//8, 1, 1)
        self.sn_value_conv = nn.Conv2d(inplanes, inplanes, 1, 1)
        self.x_mask_filters = nn.Parameter(torch.normal(
            0.0, 1.0, size=(nClass, inplanes, 1, 1)))
        self.alpha = nn.Parameter(torch.ones(1, nClass, 1, 1) * 0.1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        [b, c, h, w] = x.size()
        device = x.get_device()
        xk = self.sn_key_conv(x)
        xq = self.sn_query_conv(x)
        xv = self.sn_value_conv(x)

        x_mask = F.conv2d(x, self.x_mask_filters)
        mask_w = self.x_mask_filters.view(self.nClass, c)
        sym = torch.matmul(mask_w, torch.transpose(mask_w, 0, 1)) \
              - torch.eye(self.nClass).to(device)
        orth_loss = self.orth_lambda * torch.sum(sym)

        sampling_pos = torch.multinomial(torch.ones(h * w) * 0.5, self.nClass,
                                         replacement=True).to(device)
        sampling_pos = torch.unsqueeze(sampling_pos, 0)
        sampling_pos = torch.unsqueeze(sampling_pos, 0)
        sampling_pos = sampling_pos.expand(b, c // 8, -1)
        xk_reshaped = xk.view(b, c // 8, h * w)
        fast_filters = torch.gather(xk_reshaped, -1, sampling_pos) # b, c//8, nClass
        fast_act = torch.matmul(torch.transpose(fast_filters, 1, 2),
                                xq.view(b, c // 8, h * w))  # b, nClass, h*w
        fast_act = fast_act.view(b, self.nClass, h, w)  # b, nClass, h, w
        layout = F.softmax((torch.clamp(self.alpha, 0, 1) * fast_act +
                            x_mask) / self.kama, dim=1) # b, nClass, h, w

        layout_expand = torch.unsqueeze(layout, 1) # b, 1, nClass, h, w
        cnt = torch.sum(layout_expand, (3, 4), keepdim=True) + self.eps # b, 1, nClass, 1, 1
        xv_expand = torch.unsqueeze(xv, 2).expand(-1, -1, self.nClass, -1, -1) # b, c, nClass, h, w
        hot_area = xv_expand * layout_expand # b, c, nClass, h, w
        xv_mean = torch.mean(hot_area, (3, 4), keepdim=True) / cnt # b, c, nClass, 1, 1
        xv_std = torch.sqrt(torch.sum((hot_area - xv_mean) ** 2, [3, 4],
                                      keepdim=True) / cnt) # b, c, nClass, 1, 1
        xn = torch.sum((xv_expand - xv_mean) / (xv_std + self.eps) * layout_expand,
                       axis=2)

        x = x + self.sigma * xn
        return x, layout, orth_loss


class AttentionWeights(nn.Module):  # Attention weights for mixture norm
    def __init__(self, num_channels, k):
        super(AttentionWeights, self).__init__()
        self.k = k
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        layers = [ nn.Conv2d(num_channels, k, 1, bias=False),
                   nn.BatchNorm2d(k, eps=2e-5),
                   Hsigmoid() ]
        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)#.view(b, c)
        var = torch.var(x, dim=(2, 3)).view(b, c, 1, 1)
        y *= (var + 1e-3).rsqrt()
        #y = torch.cat((y, var), dim=1
        return self.attention(y).view(b, self.k)


class MixtureBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_channels, k, eps=2e-5, momentum=0.1, track_running_stats=True):
        super(MixtureBatchNorm2d, self).__init__(num_channels, eps=eps, momentum=momentum,
                                                 affine=False, track_running_stats=track_running_stats)
        self.k = k
        self.weight_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.bias_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.attention_weights = AttentionWeights(num_channels, k)
        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.weight_, 1, 0.1)
        nn.init.normal_(self.bias_, 0, 0.1)

    def forward(self, x):
        output = super(MixtureBatchNorm2d, self).forward(x)
        size = output.size()
        y = self.attention_weights(x) # bxk # or use output as attention input

        weight = y @ self.weight_ # bxc
        bias = y @ self.bias_ # bxc
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


