import numpy as np
import torch
from torch.autograd import Variable

# # オリジナル実装(参考)
# def mixup_process(out, reweighted_target, lam):
#     indices = np.random.permutation(out.size(0))
#     out = out*lam + out[indices]*(1-lam)
#     target_shuffled_onehot = reweighted_target[indices]
#     reweighted_target = reweighted_target * lam + target_shuffled_onehot * (1 - lam)
#     return out, reweighted_target
#
# # オリジナル実装(参考)
# def mixup_data(x, y, alpha):
#
#     '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0.:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1.
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).cuda()
#     mixed_x = lam * x + (1 - lam) * x[index,:]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

def calc_mixup_lambda(wLabel, alpha):
    if alpha > 0.0:
        if wLabel:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(1.0 + alpha, alpha)
            if lam < 0.8:
                lam = 1.0
    else:
        lam = 1.0

    return np.float32(lam)

def calc_mixup_lambdas(wLabel, alpha, batch_size):
    if alpha > 0.0:
        if wLabel:
            lam = np.random.beta(alpha, alpha, batch_size)
        else:
            lam = np.random.beta(1.0 + alpha, alpha, batch_size)
            lam[lam < 0.8] = 1.0
        lam = lam.astype(np.float32)
    else:
        lam = np.ones(batch_size, dtype=np.float32)

    return torch.FloatTensor(lam)

def mix_img(x1, x2, wLabel, alpha):

    # Compute the mixup data. Return mixed inputs and lambda

    lam = calc_mixup_lambda(wLabel, alpha)

    if lam != 1.0:
        mixed_x = lam * x1 + (1 - lam) * x2
    else:
        mixed_x = x1

    return mixed_x, lam

def rand_bbox(W, H, lam):

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_img(x1, x2, wLabel, alpha):

    # Compute the cutmix data. Return mixed inputs and lambda

    lam = calc_mixup_lambda(wLabel, alpha)

    if lam != 1.0:

        # calc bbox from lambda
        h, w, c = x1.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox(w, h, lam)

        # mix data
        mixed_x = x1
        mixed_x[bbx1:bbx2, bby1:bby2, :] = x2[bbx1:bbx2, bby1:bby2, :]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    else:
        mixed_x = x1

    return mixed_x, lam

def mixup_manifold_data(x, mix_index, mix_target, mix_lam):
    tmp_mix_lam = torch.ones_like(mix_lam)
    tmp_mix_lam[mix_index] = mix_lam[mix_index]

    # mixed_x = tmp_mix_lam * x + (1 - tmp_mix_lam) * x[mix_target, :]
    mixed_x = torch.einsum("bcwh,b->bcwh", x, tmp_mix_lam) + torch.einsum("bcwh,b->bcwh", x[mix_target, :], 1 - tmp_mix_lam)

    return mixed_x

def mixup_manifold_data_inBatch(x, alpha, y):
    # 橋本君実装
    # if alpha > 0.:
    #     lam = np.random.beta(1.0 + alpha, alpha)
    #     if lam < 0.8:
    #         lam = 1.
    # else:
    #     lam = 1.
    #
    # batch_size = x.size()[0]
    # index = torch.randperm(batch_size).cuda(device=x.get_device())
    # mixed_x = lam * x + (1 - lam) * x[index, :]
    # return mixed_x

    # 著者実装
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0.:
    #     lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 1.
    # batch_size = x.size()[0]
    # index = torch.randperm(batch_size).cuda()
    # mixed_x = lam * x + (1 - lam) * x[index,:]
    # y_a, y_b = y, y[index]
    # return mixed_x, y_a, y_b, lam

    if alpha > 0.:
        if y is not None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(1.0 + alpha, alpha)
            if lam < 0.8:
                lam = 1.
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.get_device())
    mixed_x = lam * x + (1 - lam) * x[index, :]

    if y is not None:
        y_b = y[index]
        return mixed_x, y_b, lam
    else:
        return mixed_x, None, None


def to_one_hot(inp, num_classes, wsize):

    # y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    num_of_targets = num_classes // wsize + 1
    y_onehot = torch.FloatTensor(inp.size(0), num_of_targets)

    y_onehot.zero_()
    # y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    y_onehot.scatter_(1, inp.view(-1, 1).long().cpu(), 1)

    return Variable(y_onehot.cuda(),requires_grad=False)
    # return y_onehot

