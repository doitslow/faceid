import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PSConv2d', 'WeightConv']

#=========#=========#=========#=========#=========#=========#=========#=========
class PSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, parts=4, bias=False):
        super(PSConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                dilation, dilation, groups=parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, 2 * dilation, 2 * dilation,
                                      groups=parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        # self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        self.device = torch.device("cuda",
                                   int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']))
        self.mask = torch.zeros(self.conv.weight.shape, dtype=torch.bool).to(
            self.device)
        _in_channels = in_channels // parts
        _out_channels = out_channels // parts
        for i in range(parts):
            self.mask[i * _out_channels: (i + 1) * _out_channels,
            i * _in_channels: (i + 1) * _in_channels, : , :] = True
            self.mask[(i + parts//2)%parts * _out_channels:
                      ((i + parts//2)%parts + 1) * _out_channels,
            i * _in_channels: (i + 1) * _in_channels, :, :] = True
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x_shift = self.gwconv_shift(torch.cat((x2, x1), dim=1))
        return self.gwconv(x) + self.conv(x) + x_shift


#=========#=========#=========#=========#=========#=========#=========#=========
class WeightConv(nn.Module):
    r"""Applies WeightNet to a standard convolution.

    The grouped fc layer directly generates the convolutional kernel,
    this layer has M*inp inputs, G*oup groups and oup*inp*ksize*ksize outputs.

    M/G control the amount of parameters.
    """

    def __init__(self, inp, oup, ksize, stride):
        super().__init__()

        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = nn.Conv2d(inp_gap, self.M * oup, 1, 1, 0, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.wn_fc2 = nn.Conv2d(self.M * oup, oup * inp * ksize * ksize, 1, 1, 0,
                                groups=self.G * oup, bias=False)

    def forward(self, x, x_gap):
        batch_size = x.size()[0]
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)

        if x.shape[0] == 1:  # case of batch size = 1
            x_w = x_w.reshape(self.oup, self.inp, self.ksize, self.ksize)
            x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad)
            return x

        # x = x.reshape(1, -1, x.shape[2], x.shape[3])
        # x_w = x_w.reshape(-1, self.oup, self.inp, self.ksize, self.ksize)
        # x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad, groups=x_w.shape[0])

        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        x_w = x_w.reshape(-1, self.inp, self.ksize, self.ksize)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad,
                     groups=batch_size)

        x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])
        return x

