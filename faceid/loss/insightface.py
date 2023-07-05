import torch
from torch import nn
import torch.nn.functional as F


class CosFace(nn.Module):
    def __init__(self, inplane, outplane, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(outplane, inplane))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedd, label):
        cosine = F.linear(F.normalize(embedd), F.normalize(self.weight))
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, inplane, outplane, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(outplane, inplane))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedd, label):
        cosine = F.linear(F.normalize(embedd), F.normalize(self.weight))
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine