import math

import torch
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, outputs_size, K = 2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=outputs_size,
                               out_channels=outputs_size//K,
                               kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv2 = nn.Conv2d(in_channels=outputs_size,
                               out_channels=outputs_size//K,
                               kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv3 = nn.Conv2d(in_channels=outputs_size,
                               out_channels=2,
                               kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = x[:,:,None,None]
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)

        return out


class DiscriminateLoss(nn.Module):
    def __init__(self, models, loss=nn.BCEWithLogitsLoss()):
        super(DiscriminateLoss, self).__init__()
        self.models = models
        self.loss = loss

    def forward(self, outputs, targets):
        inputs = [torch.cat((i,j),0) for i, j in zip(outputs, targets)]
        inputs = torch.cat(inputs, 1)
        batch_size = inputs.size(0)
        target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] +
                                   [[0, 1] for _ in range(batch_size//2)])
        target = target.to(inputs[0].device)
        output = self.models(inputs)
        res = self.loss(output, target)
        return res


class EmbeddLoss(nn.Module):
    def __init__(self,):
        super(EmbeddLoss, self).__init__()

    def forward(self, e_s, e_t):
        return torch.mean(torch.pow(F.normalize(e_s) - F.normalize(e_t), 2))


class Correlation(nn.Module):
    '''
	Correlation Congruence for Knowledge Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
	'''
    def __init__(self, gamma=0.4, P_order=2):
        super(Correlation, self).__init__()
        self.gamma = gamma
        self.P_order = P_order

    def forward(self, feat_s, feat_t):
        corr_mat_s = self.get_correlation_matrix(feat_s)
        corr_mat_t = self.get_correlation_matrix(feat_t)

        loss = F.mse_loss(corr_mat_s, corr_mat_t)

        return loss

    def get_correlation_matrix(self, feat):
        feat = F.normalize(feat, p=2, dim=-1)
        sim_mat = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)

        for p in range(self.P_order + 1):
            corr_mat += math.exp(-2 * self.gamma) * (2 * self.gamma) ** p / \
                        math.factorial(p) * torch.pow(sim_mat, p)

        return corr_mat


