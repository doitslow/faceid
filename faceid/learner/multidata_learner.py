import os
import sys
import logging
import random
from time import time
import math

from apex import amp
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_data_info
from data.face import FaceData
from util.others import init_random, set_deterministic, plot_loss_vs_lr
from util.metric import calc_roc, calc_top1_acc, write_roc_tprs
from util.decorate import rank0_only
from loss import build_multidata_head
from model.model_factory import build_backbone, load_backbone, MultidataModelWrapper
from timm.utils import ApexScaler
from .base_learner import BaseLearner

# from timm.optim import create_optimizer_v2, optimizer_kwargs
# from timm.optim.lv_optim_factory import create_optimizer_v2, optimizer_kwargs


class MultidataLearner(BaseLearner):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.use_partial_fc = config.use_partial_fc == 'Y'

    def get_model(self, num_classes=None, strict_load=True, pretrained=None):
        """
            KEEP in mind that different net may require different weight initialization !!!
            => check net class to ensure that weight init function is properly defined
            num_classes: determined by dataset
        """
        backbone, embed_size = build_backbone(self.config.network)
        if pretrained is not None:
            _loaded = torch.load(pretrained, map_location=self.map_loc)
            backbone = load_backbone(backbone, _loaded, strict_load)
            self.star_log('Model loaded from {}'.format(pretrained))

        if num_classes is not None:  # define classification head for training
            head = build_multidata_head(self.config.loss, embed_size, num_classes,
                                        self.wsize, self.use_partial_fc, **self.config)
            return backbone.to(self.device), head.to(self.device)
        else:
            return backbone.to(self.device)

    def gather_head(self, head):
        head_dict = {}
        for k, v in head.state_dict().items():
            gather_list = [torch.zeros_like(v) for _ in range(self.wsize)]
            dist.barrier()
            dist.all_gather(gather_list, v, group=dist.group.WORLD)
            head_dict[k] = torch.cat(gather_list, dim=0)

        return head_dict

    def load_ckpt(self, model, ckpt_path, model_only=False, optimizer=None):    # load weights for each individual process for DDP
        [backbone, head] = model
        ckpt = torch.load(ckpt_path, map_location=self.map_loc)
        backbone.module.load_state_dict(ckpt['backbone'])
        head_dict = {}
        for k, v in ckpt['partial_fc_head'].items():
            head_dict[k] = torch.chunk(v, self.wsize)[self.wrank]
        head.load_state_dict(head_dict)

        if not model_only:
            optimizer.load_state_dict(ckpt['optimizer'])
            self.epoch = ckpt['epoch']
            self.glob_step = ckpt['glob_step']
            self.best_val_acc = ckpt['best_val_acc']
            logging.info("Resume training from:\n epoch:{}\n best_val_acc:{}"
                         .format(self.epoch, self.best_val_acc))

        return model, optimizer

    @rank0_only
    def save_ckpt(self, ckpt_path, backbone, head_dict, optimizer):
        state = {
            'backbone': backbone.module.state_dict(),
            'partial_fc_head': head_dict,   # DO NOT do gather_head here, as only rank 0 will be invoked
            'optimizer': optimizer.state_dict(),
            'epoch': self.epoch,
            'glob_step': self.glob_step,
            'best_val_acc': self.best_val_acc,
        }
        torch.save(state, ckpt_path)

    def train_one_epoch(self, loaders, model, criterion, optimizer, lr_scheduler, grad_scaler):
        torch.cuda.empty_cache()
        [backbone, head] = model
        backbone.train()
        head.train()

        loader_lens = [len(loader) for loader in loaders]
        loader_ids = sum([[i] * num_batch for i, num_batch in enumerate(loader_lens)], [])
        random.seed(self.epoch)
        random.shuffle(loader_ids)
        random.seed()   # ensure that other random functions not affected
        for loader in loaders:
            loader.sampler.set_epoch(self.epoch)
        itr_loaders = [iter(loader) for loader in loaders]

        meters = {
            'Acc': self.acc_meter,
            'Loss': self.loss_meter,
        }

        for step, loader_id in enumerate(loader_ids):
            # if loader_id >= 1:
            #     weight_before = head.multi_weights[loader_id - 1].data.clone()
            # else:
            #     weight_before = head.multi_weights[loader_id + 1].data.clone()

            # -------------------------- forward --------------------------
            imgs, lbls = itr_loaders[loader_id].next()
            imgs = imgs.to(self.device)
            lbls = lbls.to(self.device).long()
            if self.precision == 'fp16':
                with torch.cuda.amp.autocast():
                    embed = backbone(imgs)
                    output = head(embed, lbls, loader_id, self.wsize, self.wrank)
            else:
                embed = backbone(imgs)
                output = head(embed, lbls, loader_id, self.wsize, self.wrank)
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]
            loss = criterion(output, lbls)  # shape of output: [img.size[0], num_class]
            dist.barrier()

            # -------------------------- backward -------------------------
            if self.precision == "fp16":

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                # grad_scaler(
                #     ce_loss, optimizer,
                #     clip_grad=self.config.clip_grad, clip_mode='norm',
                #     parameters=None,    # not in use for ApexScaler
                #     create_graph=hasattr(optimizer, 'is_second_order') and optimizer.is_second_order)
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            # optimizer.zero_grad(set_to_none=self.config.set_to_none)
            if self.config.set_to_none:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.size(0) > 10000:
                            p.grad = None

            # if loader_id >= 1:
            #     weight_after = head.multi_weights[loader_id - 1].data.clone()
            # else:
            #     weight_after = head.multi_weights[loader_id + 1].data.clone()
            # print("If weight of un-invoked FC before or after the same", weight_after==weight_before)

            # ----------------------- metrics and update ------------------
            self.glob_step += 1
            acc = calc_top1_acc(output, lbls)
            self.loss_meter.update(loss.item(), imgs.size(0))
            self.acc_meter.update(acc, imgs.size(0))
            lr = [param_group['lr'] for param_group in optimizer.param_groups]
            self.log_metric(step, lr, len(loader_ids), **meters)    # meters reset done here
            lr_scheduler.step_update(num_updates=self.glob_step, metric=self.loss_meter.avg)

        # NOTE: timm scheduler add 1 to current epoch for updating
        self.save_ckpt(self.ckpt_name + '-e_{:02d}'.format(self.epoch) + '.pth.tar',
                       backbone, self.gather_head(head), optimizer)

    def find_lr(self, loaders, model, criterion, optimizer, grad_scaler, init_value=1e-8, final_value=0.1, beta=0.98):
        [backbone, head] = model
        torch.cuda.empty_cache()
        backbone.train()
        head.train()
        loader_lens = [len(loader) for loader in loaders]
        loader_ids = sum([[i] * num_batch for i, num_batch in enumerate(loader_lens)], [])
        random.seed(self.epoch)
        random.shuffle(loader_ids)
        random.seed()   # ensure that other random functions not affected
        for loader in loaders:
            loader.sampler.set_epoch(self.epoch)
        itr_loaders = [iter(loader) for loader in loaders]

        mult = (final_value / init_value) ** (1 / (len(loader_ids) - 1))
        lr = init_value
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        avg_loss, best_loss = 0., 0.
        losses, log_lrs = [], []
        for step, loader_id in enumerate(loader_ids):
            # -------------------------- forward --------------------------
            imgs, lbls = itr_loaders[loader_id].next()
            imgs, lbls = imgs.to(self.device), lbls.to(self.device).long()
            optimizer.zero_grad()
            if self.precision == 'fp16':
                with torch.cuda.amp.autocast():
                    embed = backbone(imgs)
                    output = head(embed, lbls, loader_id, self.wsize, self.wrank)
            else:
                embed = backbone(imgs)
                output = head(embed, lbls, loader_id, self.wsize, self.wrank)
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]
            loss = criterion(output, lbls)  # shape of output: [img.size[0], num_class]
            dist.barrier()

            self.log("Step: {:d}/{:d}\t" "LR: {:.06f}\t" "Loss: {}".format(step, len(loader_ids), lr, loss.item()))
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
                grad_scaler(
                    loss, optimizer,
                    clip_grad=self.config.clip_grad, clip_mode='norm',
                    parameters=None,    # not in use for ApexScaler
                    create_graph=hasattr(optimizer, 'is_second_order') and optimizer.is_second_order)
            else:
                loss.backward()
                optimizer.step()

            lr *= mult  # Update the lr for the next step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        plot_loss_vs_lr(self.job_space, losses, log_lrs)
        return log_lrs, losses

    def train(self):
        torch.cuda.set_device(self.lrank)   # This avoids un-intentional duplicated processes on device:0
        set_deterministic(self.cudnn_benchmark)
        init_random(rank=self.wrank if self.ddp else 0)
        # ------------------------------------ 1: DATA ----------------------------------------------
        self.star_log("Loading dataset ......")
        train_data_infos, eval_data_infos = load_data_info(self.config.data_root, self.config.data_info)
        num_classes = [data_info['num_classes'] for data_info in train_data_infos]
        train_loader = [self.get_train_loader(FaceData, i) for i in train_data_infos]
        steps_per_epoch = sum(len(i) for i in train_loader)
        eval_loaders = [self.get_eval_loader(FaceData, i) for i in eval_data_infos]
        self.star_log("Completed loading dataset!")

        # ------------------------------------ 2: MODEL ----------------------------------------------
        backbone, head = self.get_model(num_classes, pretrained=self.config.pretrained)
        self.log("Using plain Cross Entropy for multidata training!")   # label smoothing is handled in head
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        optimizer = self.get_optimizer([backbone, head])
        lr_scheduler = self.get_scheduler(optimizer, steps_per_epoch)

        # ------------------------------------ 3: RUN ---------------------------------------------
        # [backbone, head], optimizer = amp.initialize([backbone, head], optimizer, opt_level='O1')
        # grad_scaler = ApexScaler()
        grad_scaler = GradScaler()
        backbone = self.model_parallel(backbone)
        if not self.config.use_partial_fc:
            head = self.model_parallel(head)

        if self.config.do_lr_search:
            self.find_lr(train_loader, [backbone, head], criterion, optimizer, grad_scaler)

        if self.resume_ckpt:
            [backbone, head], optimizer = self.load_ckpt([backbone, head], self.resume_ckpt, optimizer=optimizer)
            self.epoch += 1
            lr_scheduler.step_update(num_updates=self.glob_step, metric=0.0)
            lr_scheduler.step(self.epoch)

        self.timeit = time()
        while self.epoch < self.total_epochs:
            # self.eval(eval_loaders, backbone)
            self.train_one_epoch(train_loader, [backbone, head], criterion, optimizer, lr_scheduler, grad_scaler)
            self.eval(eval_loaders, backbone)
            self.epoch += 1
            lr_scheduler.step(self.epoch)

        if self.rank0:
            self.writer.close()
        dist.destroy_process_group()
