# =========#=========#=========#=========#=========#=========#=========#=========
import random
import numpy as np
import cv2
from PIL import Image
import os.path as op
import math

from torch.utils.data.dataset import Dataset
import torch

from data.ops.norm import generate_patch, generate_patch_fd_crop
from data.ops.align import align
from data.ops.general import color_aug, crop_proc_v2, cutoff_proc
from data.ops.mixup import mix_img, cutmix_img
from data.ops.augment import UniformAugment

def pil_to_img(src):
    return np.array(src)

def img_to_pil(src):
    return Image.fromarray(src, 'RGB')

def read_list(data_dir, list_path, part_no=-1):
    lbls = []
    img_paths = []
    rects = []
    points = []
    with open(list_path) as f:
        lines = f.readlines()
        for line in lines:
            if part_no == -1:
                lbl, img_path = line.split("\t")
                lbls.append(int(lbl))
                if data_dir != "":
                    img_paths.append(op.join(data_dir, img_path.rstrip("\n")))
                else:
                    img_paths.append(op.join(img_path.rstrip("\n")))
            else:
                line = line[:-1]
                line = line.split(" ")
                if data_dir != "":
                    img_paths.append(op.join(data_dir, line[0]))
                else:
                    img_paths.append(line[0])
                lbls.append(int(line[-1]))
                rects.append(np.array(line[1:5]).astype(int))
                point = np.array(line[6:16]).astype(int)
                point = point.astype(np.float32)
                points.append(point)
    if len(rects) == 0:
        rects = None
    if len(points) == 0:
        points = None
    return lbls, img_paths, rects, points


# =========#=========#=========#=========#=========#=========#=========#=========
class FaceData(Dataset):
    def __init__(self, info: dict, return_index=False, **kwargs):
        """
            -info: a dictionary contains property of the dataset, e.g., directory, num_classes
            -kwargs: mainly contain the setting for transforming the image
        """
        # default setting for evaluation datasets
        self.return_index = return_index

        self.scale_pixel = kwargs.get('rescale_pixel', True)
        self.auto_aug = kwargs.get('auto_aug', -1)
        self.cutoff = kwargs.get('cutoff', 3)
        self.rand_mirror = kwargs.get('rand_mirror', 1)
        self.aug_color = kwargs.get('aug_color', 0.0)
        self.rand_crop = kwargs.get('rand_crop', 1)
        self.crop_ratio = kwargs.get('crop_ratio', 0.9)
        self.mixup_type = kwargs.get('mixup_type', 0)
        self.mixup_prob = kwargs.get('mixup_prob', 0.25)
        self.mixup_alpha = kwargs.get('mixup_alpha', 0.1)
        self.part_no = kwargs.get('part_no', 0)
        self.norm_margin_ratio = kwargs.get('norm_margin_ratio', 0.1)
        self.norm_img_size = kwargs.get('norm_img_size', 112)
        self.size_in_arcface_norm = kwargs.get('size_in_arcface_norm', 112)
        self.size_w_margin = self.norm_img_size

        self.img_type = info['img_type']
        self.is_train = info['is_train']
        self._tag = info['tag']

        # Whether or not use auto augmentation
        if self.auto_aug == 0:
          self.uniaug = True
          self.uniform_policy = UniformAugment(ops_num=1)
        else:
          self.uniaug = False

        if 'normalized' in self.img_type:
            self.part_no = -1

        self.do_mix = False
        if self.is_train:
            self.do_crop = True
            if self.img_type == "original":
                self.size_w_margin = math.ceil((1.0 + self.margin_ratio) * self.norm_img_size)
            if self.mixup_type in [0, 1, 4, 5]:
                self.do_mix = True
                if self.mixup_type in [0, 4]:
                    self.mix_w_lbl = False
                else:
                    self.mix_w_lbl = True
        else:
            self.do_crop = False

        self.lbls, self.img_paths, self.rects, self.points = read_list(
            info['data_dir'], info['list_path'], self.part_no)
        self.length = len(self.lbls)

    @property
    def tag(self):

        return self._tag

    def norm_img(self, img, rect, point):
        if self.part_no == 0:
            dst = align(img, point.reshape(-1),
                        size_w_margin=self.size_w_margin,
                        size_in_ArcFaceNorm=self.size_in_arcface_norm,
                        dst_wh=self.norm_img_size)
        elif self.part_no in [1, 2, 3]:
            dst, _ = generate_patch(img, point, self.part_no, self.margin_ratio)
        elif self.part_no == 4:
            dst = generate_patch_fd_crop(img, rect, self.margin_ratio)

        if 0 < self.part_no < 5:
            normalized_img = cv2.resize(dst, (self.size_w_margin, self.size_w_margin))
            return normalized_img
        else:
            return dst

    def mix_proc(self, img, lbl):
        done_cutmix = False
        _rd = np.random.rand(1)
        if _rd < self.mixup_prob:
            index_m = random.randint(0, self.length - 1)
            img_m = cv2.imread(self.img_paths[index_m], 1)
            lbl_m = self.lbls[index_m]
            if self.part_no != -1: # Normalization is required for original img input
                img_m = self.norm_img(img_m, self.rects[index_m], self.points[index_m])
            img_m = crop_proc_v2(img_m, self.do_crop, self.img_type, 
                                 self.norm_img_size, self.rand_crop, self.crop_ratio)
            img_m = img_m[:, :, ::-1]
            img_m = img_m.astype(np.float32)

            if self.mixup_type in [0, 1]:  # mix
                img, mix_lamda = mix_img(img, img_m, self.mix_w_lbl, self.mixup_alpha)
            elif self.mixup_type in [4, 5]:  # cutmix
                img, mix_lamda = cutmix_img(img, img_m, self.mix_w_lbl, self.mixup_alpha)
                done_cutmix = True
        else:
            lbl_m = lbl
            mix_lamda = np.float32(1.0)

        return img, lbl_m, mix_lamda, done_cutmix

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index], 1)  # load BGR img
        lbl = self.lbls[index]
        img = img[:, :, ::-1]   # BGR to RGB

        if self.part_no != -1:
            img = self.norm_img(img, self.rects[index], self.points[index])

        img = crop_proc_v2(img, self.do_crop, self.img_type, self.norm_img_size,
                           self.rand_crop, self.crop_ratio)

        if self.is_train:
            if self.uniaug: # NOT FOR EVAL
              img = pil_to_img(self.uniform_policy(img_to_pil(img)))
            img = img.astype(np.float32)
            h, w, c = img.shape
            assert h == w

            # mix or cutmix
            if self.do_mix: # NOT FOR EVAL
                img, lbl_m, mix_lamda, done_cutmix = self.mix_proc(img, lbl)
            else:
                done_cutmix = False

            if self.aug_color:  # NOT FOR EVAL
                img = color_aug(img, self.aug_color)

            if self.rand_mirror:    # NOT FOR EVAL
                _rd = random.randint(0, 1)
                if _rd == 1:
                    img = img[:, ::-1, :]

            if not done_cutmix and self.cutoff > 0: # NOT FOR EVAL
                img = cutoff_proc(img, self.cutoff)

        if self.scale_pixel:
            img = img - 127.5
            img = img * 0.0078125

        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img.copy()).float()

        if self.do_mix and self.mix_w_lbl:
            return img_tensor, lbl, lbl_m, mix_lamda

        if self.return_index:
            return img_tensor, lbl, index
        else:
            return img_tensor, lbl

    def __len__(self):
        return self.length
