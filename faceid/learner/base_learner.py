import os
import sys
from time import time
import random
import logging
import math
from typing import Union, List
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.others import RingLoss, LabelSmoothLoss
from data import load_data_info
from data.face import FaceData
from util.others import init_random, set_deterministic, plot_loss_vs_lr
from util.others import AverageMeter as AvgMeter
from util.decorate import rank0_only
from util.dist_ops import do_all_gather
from util.metric import calc_roc, calc_top1_acc, write_roc_tprs
# from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
# from timm.utils import ApexScaler, NativeScaler
from model import build_backbone, load_backbone, ModelWrapper
from loss.loss_factory import build_head
from .abstract_learner import AbstractLearner
from .learn_helpers import calc_score

from timm.optim.lv_optim_factory import create_optimizer_v2, optimizer_kwargs
# =========#=========#=========#=========#=========#=========#=========#=========


class BaseLearner(AbstractLearner):
    """
    Think about what should be the attribute of learner?
        - status of learning: e.g., epoch, accuracy, ....
          They are alawys needed in the same way for different training

    Think about what should NOT be the attribute of learner?
        - components required for learning: e.g., model, optmizier.
          They are needed differently for each different training

    """
    def __init__(self, config, training=True, *args, **kwargs):  # this make the code short, but very ungeneralized
        super().__init__(config)
        self.log_args(self.config)  # log all paramters that has been passed into learner

        if training:
            self.ckpt_name = os.path.join(self.job_space, 'ckpt-' + os.path.basename(self.job_space))
            self.print_freq = config.print_freq
            self.total_epochs = config.epochs
            self.resume_ckpt = config.resume_ckpt
            self.precision = config.precision
            self.cudnn_benchmark = config.cudnn_benchmark == 'Y'
            if self.rank0:
                self.writer = SummaryWriter(self.job_space)

            self.epoch = 0
            self.glob_step = 0
            self.best_val_acc = []
            self.acc_meter, self.loss_meter = AvgMeter(), AvgMeter()
            self.timeit = time()

    def get_train_loader(self, dataset_class: Dataset, data_info: dict) -> DataLoader:
        self.log("Loading training dataset '{}' with class {}".format(
            data_info['tag'] if 'tag' in data_info.keys() else "missing tag", dataset_class))
        dataset = dataset_class(data_info, **self.config)
        sampler = DistributedSampler(dataset, num_replicas=self.wsize, rank=self.wrank)
        loader = DataLoader(
            dataset, batch_size=self.loader_batch, pin_memory=True,
            num_workers=4, drop_last=True, sampler=sampler
        )

        return loader

    def get_eval_loader(self, dataset_class: Dataset, data_info: dict) -> DataLoader:
        self.log("Loading evaluation dataset '{}' with class {}".format(
            data_info['tag'] if 'tag' in data_info.keys() else "missing tag", dataset_class))
        dataset = dataset_class(data_info, return_index=True, **self.config)
        sampler = DistributedSampler(dataset, num_replicas=self.wsize, rank=self.wrank)
        loader = DataLoader(
            dataset, batch_size=self.loader_batch, pin_memory=True,
            num_workers=4, drop_last=False, sampler=sampler
        )

        return loader

    def get_model(self, num_classes: Union[int, list] = None, strict_load=True, pretrained=None):
        """
            KEEP in mind that different net may require different weight initialization !!!
            => check net class to ensure that weight init function is properly defined
            num_classes: determined by dataset

            if num_classes is None, create backbone only
        """
        backbone, embed_size = build_backbone(self.config.network)
        if num_classes is not None: # define classification head for training
            head = build_head(self.config.loss, embed_size, num_classes, **self.config)
            model = ModelWrapper(backbone, head)
        else:   # define the backbone only for evaluation
            model = backbone
        model.embed_size = embed_size

        if pretrained is not None and num_classes is None:  # only used for evaluation
            _loaded = torch.load(pretrained, map_location=self.map_loc)
            model = load_backbone(model, _loaded, strict_load)
            self.star_log('Model loaded from {}'.format(pretrained))

        return model.to(self.device)

    def get_criterion(self, label_smoothing):
        if label_smoothing > 0.0:
            self.log("Using Label Smoothign criterion!")
            criterion = LabelSmoothLoss(label_smoothing).to(self.device)
        else:
            self.log("Using Cross Entropy!")
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(self.device)

        return criterion

    def get_optimizer(self, model, **kwargs):
        self.star_log("Using timm optimizer: {}".format(self.config.opt))
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=self.config))
        self.log(optimizer)

        return optimizer

    def get_scheduler(self, optimizer, steps_per_epoch):
        self.star_log("Using timm scheduler: {}".format(self.config.sched))
        lr_scheduler, _ = create_scheduler(self.config, optimizer, steps_per_epoch)

        return lr_scheduler

    # In DDP: need to load model weights for each individual process
    def load_ckpt(self, model, ckpt_path, model_only=False, optimizer=None):
        ckpt = torch.load(ckpt_path, map_location=self.map_loc)
        model.module.load_state_dict(ckpt['model'], strict=False)

        if not model_only:
            optimizer.load_state_dict(ckpt['optimizer'])
            self.epoch = ckpt['epoch']
            self.glob_step = ckpt['glob_step']
            self.best_val_acc = ckpt['best_val_acc']
            logging.info("Resume training from:\n epoch:{}\n best_val_acc:{}"
                         .format(self.epoch, self.best_val_acc))
            return model, optimizer

        return model

    @rank0_only
    def save_ckpt(self, ckpt_path, model, optimizer):
        state = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': self.epoch,
            'glob_step': self.glob_step,
            'best_val_acc': self.best_val_acc,
        }
        torch.save(state, ckpt_path)

    @rank0_only
    def log_metric(self, step, lr, steps_per_epoch, **kwargs):
        if self.glob_step % self.print_freq == 0:
            speed = self.batch_size * self.print_freq / (time() - self.timeit)
            if not isinstance(lr, list):
                msg = "Epoch: [{0}][{1}/{2}]\t" "Speed: {3:.2f}imgs/sec\t" "LR: {4:.7f}\t"\
                    .format(self.epoch, step, steps_per_epoch, speed, lr)
            else:
                msg = "Epoch: [{0}][{1}/{2}]\t" "Speed: {3:.2f}imgs/sec\t"\
                    .format(self.epoch, step, steps_per_epoch, speed)
                for i, lr_item in enumerate(lr):
                    msg += "LR_{0}: {1:.7f}\t".format(i, lr_item)
            for name, meter in kwargs.items():
                msg += "{}: {:.6f} ({:.6f})\t".format(name, meter.val, meter.avg)
                self.writer.add_scalar(name, meter.avg, self.glob_step)
                meter.reset()
            self.log(msg)
            self.timeit = time()

    def calc_score(self, model, loader):
        dist.barrier()
        model.eval()
        torch.cuda.empty_cache()

        score, label = None, None
        if self.rank0:
            all_embeds, all_lbls, all_inds = [], [], []
        with torch.no_grad():
            for batch_idx, (img, lbl, ind) in enumerate(loader):
                print("Calculating similarity score {}/{}".format(batch_idx, len(loader)))
                img, lbl, ind = img.to(self.device), lbl.to(self.device), ind.to(self.device)
                embed = model(img)
                dist.barrier()
                embeds = do_all_gather(embed, self.wsize)
                lbls = do_all_gather(lbl, self.wsize)
                inds = do_all_gather(ind, self.wsize)

                if self.rank0:
                    all_embeds.append(embeds)
                    all_inds.append(inds.cpu().numpy())
                    all_lbls.append(lbls.cpu().numpy())

            if self.rank0:
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

    def infer(self, model_pth=None, model=None, eval_loaders: list = None):
        torch.cuda.set_device(self.lrank)
        if model_pth is None:
            model_pth = self.config.model_path

        if eval_loaders is None:
            _, eval_data_infos = load_data_info(self.config.data_root, self.config.data_info)
            eval_loaders = [self.get_eval_loader(FaceData, i) for i in eval_data_infos]

        if model is None:
            model = self.get_model(self.config.network, pretrained=model_pth)
        else:
            _loaded = torch.load(model_pth, map_location=self.map_loc)
            print(_loaded.keys())
            model.load_state_dict(_loaded['net'], strict=True)
            # model = load_backbone(model, _loaded, strict_load=True)

        model = model.to(self.device)
        model = self.model_parallel(model)
        self.eval(eval_loaders, model, ckpt_name=model_pth.replace('.pth.tar', ''))

    def eval(self, loaders, model, ckpt_name=None):
        if ckpt_name is None:
            ckpt_name = self.ckpt_name + '-e_{:02d}'.format(self.epoch)
        for loader in loaders:
            tag = loader.dataset.tag if hasattr(loader.dataset, 'tag') else 'missing_tag'
            # score, label = self.calc_score(model, loader)
            score, label = calc_score(model, loader, self.wsize, self.device, self.rank0)
            if self.rank0:
                froc = ckpt_name.replace('ckpt-', '') + '-{}-roc.csv'.format(tag)
                ftprs = froc.replace('roc', 'tprs')
                fpr, tpr, tprs, AUC = calc_roc(score, label)
                write_roc_tprs(froc, ftprs, fpr, tpr, tprs, AUC)
                self.star_log('TPR @ FPR=(E-7 to E-1) for {}:\n{}'
                              .format(tag, ['{:.6f}'.format(i) for i in tprs]))
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('{}_E-5'.format(tag), tprs[2], self.epoch)

    #FIXME: how to handle gradient explosion
    def find_lr(self, loader, model, criterion, optimizer, grad_scaler, init_value=1e-8, final_value=10., beta=0.98):
        mult = (final_value / init_value) ** (1 / (len(loader) - 1))
        lr = init_value
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        avg_loss, best_loss = 0., 0.
        losses, log_lrs = [], []
        model.train()
        for step, (imgs, lbls) in enumerate(loader):
            imgs, lbls = imgs.to(self.device), lbls.to(self.device).long()
            optimizer.zero_grad()
            if self.precision == 'fp16':
                with torch.cuda.amp.autocast():
                    output = model(imgs, lbls)
            else:
                output = model(imgs, lbls)
            loss = criterion(output, lbls)

            self.log("Step: {:d}/{:d}\t" "LR: {:.06f}\t" "Loss: {}".format(step, len(loader), lr, loss.item()))
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (step + 1))  # Compute the smoothed loss
            if step > 0 and loss.item() > 45:
                plot_loss_vs_lr(self.job_space, losses, log_lrs)
                return log_lrs, losses
            if step > 0 and smoothed_loss > 4 * best_loss:  # Stop if the loss is exploding
                plot_loss_vs_lr(self.job_space, losses, log_lrs)
                return log_lrs, losses
            if smoothed_loss < best_loss or step == 0:  # Record the best loss
                best_loss = smoothed_loss
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # -------------------------- backward -------------------------
            if self.precision == "fp16":
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr *= mult  # Update the lr for the next step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        plot_loss_vs_lr(self.job_space, losses, log_lrs)
        return log_lrs, losses

    def train_one_epoch(self, loader: Union[DataLoader, List[DataLoader]], model, criterion, optimizer, lr_scheduler, grad_scaler):
        torch.cuda.empty_cache()
        model.train()
        meters = {
            'Acc': self.acc_meter,
            'Loss': self.loss_meter,
        }
        loader.sampler.set_epoch(self.epoch)

        for step, (imgs, lbls) in enumerate(loader):
            # -------------------------- forward --------------------------
            imgs = imgs.to(self.device)
            lbls = lbls.to(self.device).long()
            if self.precision == 'fp16':
                with torch.cuda.amp.autocast():
                    output = model(imgs, lbls)
            else:
                output = model(imgs, lbls)
            assert not (isinstance(output, tuple) or isinstance(output, list)), \
                'Multiple items produced from forward propagation'
            loss = criterion(output, lbls)  # shape of output: [img.size[0], num_class]
            dist.barrier()
            # -------------------------- backward -------------------------
            if self.precision == "fp16":
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=self.config.set_to_none)   # should follow optimizer step
            # ----------------------- metrics and update ------------------
            self.glob_step += 1
            acc = calc_top1_acc(output, lbls)
            self.loss_meter.update(loss.item(), imgs.size(0))
            self.acc_meter.update(acc, imgs.size(0))
            lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            self.log_metric(step, lrs, len(loader), **meters)    # meters reset done here
            lr_scheduler.step_update(num_updates=self.glob_step, metric=self.loss_meter.avg)

        self.save_ckpt(self.ckpt_name + '-e_{:02d}'.format(self.epoch) + '.pth.tar', model, optimizer)

    def train(self):
        torch.cuda.set_device(self.lrank)   # This avoids un-intentional duplicated processes on device:0
        set_deterministic(self.cudnn_benchmark)
        init_random(rank=self.wrank if self.ddp else 0)
        # ------------------------------------ 1: DATA ----------------------------------------------
        self.star_log("Loading dataset ......")
        train_data_infos, eval_data_infos = load_data_info(self.config.data_root, self.config.data_info)
        assert len(train_data_infos) == 1, 'Base learner does not deal multiple training datasets'
        num_classes = train_data_infos[0]['num_classes']
        loader = self.get_train_loader(FaceData, train_data_infos[0])
        steps_per_epoch = len(loader)
        eval_loaders = [self.get_eval_loader(FaceData, i) for i in eval_data_infos]
        self.star_log("Completed loading dataset!")

        # ------------------------------------ 2: MODEL ----------------------------------------------
        model = self.get_model(num_classes, pretrained=self.config.pretrained)
        criterion = self.get_criterion(self.config.label_smoothing)
        optimizer = self.get_optimizer(model)
        lr_scheduler = self.get_scheduler(optimizer, steps_per_epoch)
        grad_scaler = None
        if self.precision == 'fp16':
            grad_scaler = GradScaler()
        model = self.model_parallel(model)

        # ------------------------------------ 3: RUN ---------------------------------------------
        if self.config.do_lr_search:
            self.find_lr(loader, model, criterion, optimizer, grad_scaler)
        if self.resume_ckpt:
            model, optimizer = self.load_ckpt(model, self.resume_ckpt, optimizer=optimizer)
            self.epoch += 1
            lr_scheduler.step_update(num_updates=self.glob_step, metric=0.0)
            lr_scheduler.step(self.epoch)
        self.timeit = time()
        while self.epoch < self.total_epochs:
            self.eval(eval_loaders, model)
            self.train_one_epoch(loader, model, criterion, optimizer, lr_scheduler, grad_scaler)
            # self.eval(eval_loaders, model)
            self.epoch += 1
            lr_scheduler.step(self.epoch)
        if self.rank0:
            self.writer.close()
        dist.destroy_process_group()