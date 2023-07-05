from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

__all__ = ['Arcface', 'Softmax', 'MdArcface', 'SdArcface', 'SdSoftmax',
           'SdHybrid', 'MdSoftmax', 'AttnArcFace', 'MarginDistill',
           'AttnTranArcface', 'ArcfaceExtra', 'ArcfaceRkd', 'ArcfaceEmbd']

# ===============================================================================
class Arcface(nn.Module):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(Arcface, self).__init__()
        self.emb_size = emb_size
        self.num_class = num_class
        self.s = kwargs.get('margin_s')
        self.m = kwargs.get('margin_m')
        self.easy_margin = kwargs.get('easy_margin')
        self.use_sd = kwargs.get('use_sd') == 'Y'
        assert self.s > 0.0
        assert self.m >= 0.0
        assert self.m < (math.pi / 2)
        self.model = model
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.def_param()

    def def_param(self):
        self.weight = Parameter(torch.FloatTensor(self.num_class, self.emb_size))
        nn.init.xavier_uniform_(self.weight)

    def compute(self, fea, weight, label):
        cosine = F.linear(F.normalize(fea), F.normalize(weight),
                          bias=None).float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=cosine.get_device())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def forward(self, label, image):
        embedd = self.model(image)
        if label is None:
            return embedd
        return self.compute(embedd, self.weight, label)


class MdArcface(Arcface):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(MdArcface, self).__init__(model, emb_size, num_class, **kwargs)

    def def_params(self):
        self.weight1 = Parameter(torch.FloatTensor(self.num_class, self.emb_size))
        self.weight2 = Parameter(torch.FloatTensor(self.num_class, self.emb_size))
        self.weight3 = Parameter(torch.FloatTensor(self.num_class, self.emb_size))
        self.weight4 = Parameter(torch.FloatTensor(self.num_class, self.emb_size))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.xavier_uniform_(self.weight3)
        nn.init.xavier_uniform_(self.weight4)

    def forward(self, label, image, weight=None):
        if not weight:
            [fea1, fea2, fea3, embedd], [outfeat1, outfeat2, outfeat3, outfeat4] \
                = self.model(image)
            if label is None:
                return embedd
            out1 = self.compute(fea1, self.weight1, label)
            out2 = self.compute(fea2, self.weight2, label)
            out3 = self.compute(fea3, self.weight3, label)
            out4 = self.compute(embedd, self.weight4, label)
            return [out1, out2, out3, out4], [outfeat1, outfeat2, outfeat3, outfeat4]
        else:
            embedd = self.model(image, weight)
            out = self.compute(embedd, weight['faceid.weight4'], label)
            return out


class SdArcface(MdArcface):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(SdArcface, self).__init__(model, emb_size, num_class, **kwargs)

    def forward(self, label, image):
        fea1, fea2, fea3, fea4 = self.model(image)
        if label is None:
            return fea4
        out1 = self.compute(fea1, self.weight1, label)
        out2 = self.compute(fea2, self.weight2, label)
        out3 = self.compute(fea3, self.weight3, label)
        out4 = self.compute(fea4, self.weight4, label)
        return [out1, out2, out3, out4], [fea1, fea2, fea3, fea4]


class SdHybrid(Arcface):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(SdHybrid, self).__init__(model, emb_size, num_class, **kwargs)
        self.fc1 = nn.Linear(emb_size, num_class)
        self.fc2 = nn.Linear(emb_size, num_class)
        self.fc3 = nn.Linear(emb_size, num_class)

    def forward(self, label, image):
        fea1, fea2, fea3, fea4 = self.model(image)
        if label is None:
            return fea4

        out1 = self.fc1(fea1.float()).float()
        out2 = self.fc2(fea2.float()).float()
        out3 = self.fc3(fea3.float()).float()
        out4 = self.compute(fea4, self.weight, label)
        return [out1, out2, out3, out4], [fea1, fea2, fea3, fea4]


class MarginDistill(nn.Module):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(MarginDistill, self).__init__()
        self.emb_size = emb_size
        self.num_class = num_class
        self.s = kwargs.get('margin_s')
        self.m = kwargs.get('margin_m')
        self.easy_margin = kwargs.get('easy_margin')
        self.use_sd = kwargs.get('use_sd') == 'Y'
        assert self.s > 0.0
        assert self.m >= 0.0
        assert self.m < (math.pi / 2)
        self.model = model
        self.t_weight = Parameter(torch.FloatTensor(num_class, emb_size))
        self.s_weight = Parameter(torch.FloatTensor(num_class, emb_size))
        nn.init.xavier_uniform_(self.s_weight)
        nn.init.xavier_uniform_(self.t_weight)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.Mmax = 0.5
        self.Mmin = 0.2

    def cal_m(self, label, embedd):
        gt_one_hot = F.one_hot(label, num_classes=self.num_class).float()
        teacher_centers = torch.matmul(gt_one_hot, F.normalize(self.t_weight))
        el = F.normalize(embedd) * teacher_centers
        cos_loss = torch.sum(el, -1)
        Dmax = torch.max(cos_loss)
        margin = torch.mul((self.Mmax - self.Mmin) / Dmax, cos_loss) + self.Mmin
        return margin

    def forward(self, label, image):
        embedd = self.model(image)
        if label is None:
            return embedd

        self.m = self.cal_m(label, embedd).unsqueeze(-1)
        self.th = torch.cos(math.pi - self.m)
        self.mm = torch.sin(math.pi - self.m) * self.m

        cosine = F.linear(F.normalize(embedd), F.normalize(self.s_weight),
                          bias=None).float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * torch.cos(self.m) - sine * torch.sin(self.m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=cosine.get_device())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class AttnArcFace(Arcface):
    def __int__(self, model, emb_size, num_class, **kwargs):
        super(AttnArcFace, self).__init__(model, emb_size, num_class, **kwargs)

    def forward(self, label, image):
        embedd, orth_loss = self.model(image)
        if label is None:
            return embedd
        return self.compute(embedd, self.weight, label), orth_loss


class Softmax(nn.Module):
    """
    nn.CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    => such that we only need 1 FC layer here
    """
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(Softmax, self).__init__()
        self.model = model
        self.fc = nn.Linear(emb_size, num_class)

    def forward(self, label, image):
        embedd = self.model(image)
        if label is None:
            return embedd
        return self.fc(embedd)


class SdSoftmax(nn.Module):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(SdSoftmax, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(emb_size, num_class)
        self.fc2 = nn.Linear(emb_size, num_class)
        self.fc3 = nn.Linear(emb_size, num_class)
        self.fc4 = nn.Linear(emb_size, num_class)

    def forward(self, label, image):
        fea1, fea2, fea3, fea4 = self.model(image)
        if label is None:
            return fea4

        out1 = self.fc1(fea1.float()).float()
        out2 = self.fc2(fea2.float()).float()
        out3 = self.fc3(fea3.float()).float()
        out4 = self.fc4(fea4.float()).float()
        return [out1, out2, out3, out4], [fea1, fea2, fea3, fea4]


class MdSoftmax(nn.Module):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(MdSoftmax, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(emb_size, num_class)
        self.fc2 = nn.Linear(emb_size, num_class)
        self.fc3 = nn.Linear(emb_size, num_class)
        self.fc = nn.Linear(emb_size, num_class)

    def forward(self, label, image, weight=None):
        if not weight:
            [fea1, fea2, fea3, embedd], [outfeat1, outfeat2, outfeat3, outfeat4] \
                = self.model(image)
            if label is None:
                return embedd
            out1 = self.fc1(fea1.float()).float()
            out2 = self.fc2(fea2.float()).float()
            out3 = self.fc3(fea3.float()).float()
            out4 = self.fc(embedd.float()).float()
            return [out1, out2, out3, out4], [outfeat1, outfeat2, outfeat3, outfeat4]
        else:
            embedd = self.model(image, weight)
            out = F.linear(embedd,
                           weight['faceid.fc.weight'],
                           weight['faceid.fc.bias'],).float()
            return out


class AttnTranArcface(Arcface):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(AttnTranArcface, self).__init__(model, emb_size, num_class, **kwargs)

    def forward(self, label, image):
        embedd, feat_list = self.model(image)
        if label is None:
            return embedd
        return self.compute(embedd, self.weight, label), feat_list


class ArcfaceExtra(Arcface):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(ArcfaceExtra, self).__init__(model, emb_size, num_class, **kwargs)

    def forward(self, label, image):
        embedd, feat_list = self.model(image)
        if label is None:
            return embedd
        return embedd, feat_list, self.compute(embedd, self.weight, label)


class ArcfaceRkd(Arcface):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(ArcfaceRkd, self).__init__(model, emb_size, num_class, **kwargs)

    def forward(self, label, image):
        embedd = self.model(image)
        if label is None:
            return embedd
        return embedd


class ArcfaceEmbd(Arcface):
    def __init__(self, model, emb_size, num_class, **kwargs):
        super(ArcfaceEmbd, self).__init__(model, emb_size, num_class, **kwargs)

    def forward(self, label, image):
        embedd = self.model(image)
        if label is None:
            return embedd
        return embedd, self.compute(embedd, self.weight, label)