from os.path import join
import csv
import math
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributed as dist

def plot_loss_vs_lr(out_dir, losses, log_lrs):
    with open(join(out_dir, 'loss_vs_lr.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(losses)
        writer.writerow(log_lrs)
        csvfile.close()

    plt.figure()
    plt.plot(log_lrs, losses, label='ce_loss')
    plt.savefig(join(out_dir, 'loss_vs_lr.jpg'))

def do_all_gather(tensor, wsize):
    gather_list = [torch.zeros_like(tensor) for _ in range(wsize)]
    dist.barrier()
    torch.distributed.all_gather(gather_list, tensor)
    gathered_tensor = torch.cat(gather_list, dim=0)

    return gathered_tensor

def prorate_lr(lrs: list, batch_size: int, base: int = 512):
    new_lrs = []
    for lr in lrs:
        new_lrs.append(float(lr * batch_size / float(base)))

    return new_lrs

def calc_score(model, loader, wsize, device, rank0):
    dist.barrier()
    model.eval()
    torch.cuda.empty_cache()

    score, label = None, None
    if rank0:
        all_embeds, all_lbls, all_inds  = [], [], []
    with torch.no_grad():
        for batch_idx, (img, lbl, ind) in enumerate(loader):
            print("Calculating similarity score {}/{}".format(batch_idx, len(loader)))
            img, lbl, ind = img.to(device), lbl.to(device), ind.to(device)
            embed = model(img)
            dist.barrier()
            embeds = do_all_gather(embed, wsize)
            lbls = do_all_gather(lbl, wsize)
            inds = do_all_gather(ind, wsize)

            if rank0:
                all_embeds.append(embeds)
                all_inds.append(inds.cpu().numpy())
                all_lbls.append(lbls.cpu().numpy())

        if rank0:
            all_embeds = torch.cat(all_embeds, axis=0)
            all_lbls = np.concatenate(all_lbls, axis=0)
            all_inds = np.concatenate(all_inds, axis=0)
            # Remove extra
            uniqs, uniqs_inds = np.unique(all_inds, return_index=True)
            label = all_lbls[uniqs_inds.squeeze()]
            assert len(label) == len(loader.dataset), 'Something wrong with unique label selection!'
            uniq_embeds = all_embeds[uniqs_inds.squeeze(), :]
            uniq_embeds = F.normalize(uniq_embeds)
            score = F.linear(uniq_embeds, uniq_embeds, bias=None)
            del all_embeds
            del uniq_embeds
            score = score.cpu().numpy()
            score = score[np.triu_indices(label.shape[0], k=1)]

    dist.barrier()
    torch.cuda.empty_cache()

    return score, label
