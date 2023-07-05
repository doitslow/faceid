import csv
import os.path as op
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import torch
import torch.nn.functional as F

__all__ = ['calc_roc', 'calc_top1_acc', 'write_roc_tprs', 'roc_curve_fast']

def roc_curve_fast(ans, score, num_thresh=1000):
    # スケーリング ⇒ int化
    score = score * num_thresh
    g_pair = np.sort(score[ans == 1].astype(np.int16))
    i_pair = np.sort(score[ans != 1].astype(np.int16))

    # ユニークなスコアとその出現回数の取得
    g_uniq, g_count = np.unique(g_pair, return_counts=True)
    i_uniq, i_count = np.unique(i_pair, return_counts=True)

    # 本来の要素数を確認
    gnum = g_pair.shape[0]
    inum = i_pair.shape[0]

    # 閾値の配列を作成
    # FPR, TPRの配列を初期化
    min_score = min(np.amin(g_uniq), np.amin(i_uniq))
    thresh = np.arange(min_score, num_thresh)
    num_thresh = len(thresh)
    fpr = np.zeros_like(thresh).astype(np.float32)
    tpr = np.zeros_like(thresh).astype(np.float32)

    # ROCカーブの作成
    for i, th in enumerate(thresh):
        tpr[i] = np.sum(g_count[g_uniq > th])
        fpr[i] = np.sum(i_count[i_uniq > th])
    i_scale = np.float32(1.0 / float(inum))
    g_scale = np.float32(1.0 / float(gnum))
    return fpr[::-1] * i_scale, tpr[::-1] * g_scale, thresh[::-1]

def calc_roc(score, label):
    """Calculate the entire roc curve as well as interested tpr values"""
    # Make issame matrix
    ans = np.zeros((label.shape[0], label.shape[0]), dtype=int)
    for i in range(label.shape[0]):
        ans[i, :] = (label == label[i])
    ansMod = np.array(ans)
    ansMod = ansMod[np.triu_indices(label.shape[0], k=1)]  # extract upper triangular

    # fpr, tpr, threshold = roc_curve(ansMod, score)
    fpr, tpr, threshold = roc_curve_fast(ansMod, score)
    del score
    AUC = auc(fpr, tpr)
    # Check tpr on focused FPR
    fpr_ths = [1.0E-7, 1.0E-6, 1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2, 1.0E-1]
    def get_best_tpr(fpr_th):
        dif = np.abs(fpr - fpr_th)
        idxs = np.where(dif == dif.min())
        best_tpr = 0
        for x in idxs[0]:
            if best_tpr < tpr[x]:
                best_tpr = tpr[x]
        return best_tpr
    tprs = [get_best_tpr(fpr_th)*100 for fpr_th in fpr_ths]
    return fpr, tpr, tprs, AUC

def calc_top1_acc(output, label):
    assert (output.size()[0] == label.size()[0])
    pred_label = output.argmax(dim=1)
    count = (label == pred_label).sum()
    return count.item() / label.size()[0]

def write_roc_tprs(roc_file, tprs_file, fpr, tpr, tprs, AUC):
    with open(roc_file, 'w') as f:
        writer = csv.writer(f)
        for item in [["FPR"] + list(fpr), ["TPR"] + list(tpr)]:
            writer.writerow(item)
        f.close()
    with open(tprs_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'AUC', 'E-7', 'E-6', 'E-5', 'E-4',
                         'E-3', 'E-2', 'E-1'])
        writer.writerow([op.basename(roc_file.strip('-roc.csv')),
                         '{:.8f}'.format(AUC)]
                        + ['{:.4f}'.format(i) for i in tprs])
        f.close()