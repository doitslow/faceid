import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

class RingLoss(nn.Module):
    def __init__(self, device, type=0, loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        # self.radius = Parameter(torch.Tensor(1))
        self.radius = Parameter(torch.FloatTensor(1))   # こいつも学習するのが肝
        self.radius.data.fill_(1.0)
        self.loss_weight = loss_weight
        self.type = type
        self.device = device

    def forward(self, x):

        # calculate L2 norm of feature
        # x = x.pow(2).sum(dim=1).pow(0.5)
        x = torch.norm(x, dim=1)

        # Initialize the radius with the mean feature norm of first iteration
        # if self.radius.data[0] < 0:
        #     self.radius.data.fill_(x.mean().data)

        if self.type == 0: # L2 Loss (たぶん論文通りの実装), if not specified
            # print("self.radius", self.radius)

            diff = x.sub(self.radius.expand_as(x))
            # diff_sq = torch.pow(torch.abs(diff), 2).mean()
            diff_sq = torch.pow(diff, 2.0).mean() * 0.5

            ringloss = diff_sq.mul_(self.loss_weight)
        elif self.type == 1: # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        elif self.type == 2: # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2

        return ringloss

class LabelSmoothLoss(nn.Module):
    # https://github.com/Boyiliee/MoEx/blob/dbc88432e6007693ce3022e2d95b59d95034b508/ImageNet/main_moex.py#L657

    def __init__(self, smoothing=0.0, reduction="mean"):
        # ignore_indexは未実装

        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        if self.reduction == "mean":
            loss = (-weight * log_prob).sum(dim=-1).mean()
        elif self.reduction == "none":
            loss = (-weight * log_prob).sum(dim=-1)

        return loss


def loss_kd_regularization(outputs, labels, params):
    """
    https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation/blob/master/my_loss_function.py
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = params.reg_alpha
    T = params.reg_temperature
    correct_prob = 0.99    # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i ,labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1))*params.multiplier

    KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

    return KD_loss