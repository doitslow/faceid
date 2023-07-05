import numpy as np
import cv2
import random

def saturation_aug(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray_sum = np.sum(gray, axis=2, keepdims=True)
    gray_sum *= (1.0 - alpha)

    src *= alpha
    src += gray_sum
    return src

def brightness_aug(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    src *= alpha
    return src


def color_aug(img, x):
    augs = [brightness_aug]
    for aug in augs:
        _rd = random.randint(0, 1)
        if _rd == 1:
            img = aug(img, x)
    return img


def scale_down(src_size, size):
    """Scales down crop size if it's larger than image size.
    """
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w * sh) / h, sh
    if sw < w:
        w, h = sw, float(h * sw) / w
    return int(w), int(h)


def fixed_crop(src, x0, y0, w, h, size=None):
    # 参照だとtorch.tensorへのコピーでこける
    out = src[y0:y0 + h, x0:x0 + w, :].copy()
    return out

def random_crop(src, size):
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)
    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)
    out = fixed_crop(src, int(x0), int(y0), new_w, new_h, size)
    return out

def center_crop(src, size):
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)
    x0 = (w - new_w) / 2.0
    y0 = (h - new_h) / 2.0
    out = fixed_crop(src, int(x0), int(y0), new_w, new_h, size)
    return out

# 20200915時点の実装
def crop_proc(img, part_no, margin_ratio, norm_img_size, rand_crop, crop_ratio, h, w):
    # Random Crop(training)のケースと，Center Crop("主に"validation)のケースあり
    # 入力画像は下記3パターン
    #   A. オリジナル
    #   B. 正規化 (※ 正規化はしてあるけどCropするInsight Face 仕様なので，正確にはマージン付与)
    #   C. 正規化 + マージン付与
    if part_no != -1:  # Aのパターン
        # Aのパターンでは，正規化画像を作る際にマージン付与の有無を指定する
        # そのため，validationではマージン付与を指定しないことで，Cropなしで評価可能である
        # ⇒ 学習の場合だけにフォーカス
        if margin_ratio != 0.0:
            # print ("original img.shape: ", img.shape)
            # cropするサイズは，norm_img_sizeで固定なのでいちいち計算しない
            _size = (norm_img_size, norm_img_size)
            img = random_crop(img, _size)
            # random_cropからコールしているfixed_cropを参照から代入に変えたので不要に
            # img = cv2.resize(img, (self.norm_img_size, self.norm_img_size), interpolation=cv2.INTER_LINEAR)
    else:  # B or C のパターン
        # Crop周辺のパラメタが混雑している & 思想がぐちゃぐちゃなので分岐が分かりにくい
        # 将来的に整備したい
        if crop_ratio != 1.0:  # Cropする場合
            if h == norm_img_size:  # Bのパターン(のはず)
                # _size = (int(h * self.crop_ratio), int(w * self.crop_ratio))
                # if self.rand_crop:          # ケース 1
                #    img = random_crop(img, _size)
                # else:                       # ケース 2
                #    img = center_crop(img, _size)
                img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
            else:
                if rand_crop:  # ケース 3
                    # Crop率を可変にしている
                    range = np.random.rand() * abs(1.0 - crop_ratio) * 2 + 1.0 - abs(1.0 - crop_ratio)
                    _size = (int(norm_img_size * range), int(norm_img_size * range))
                    img = random_crop(img, _size)
                    # Cropされた画像がnorm_img_sizeとは限らないのでリサイズが必要
                    img = cv2.resize(img, (norm_img_size, norm_img_size), interpolation=cv2.INTER_LINEAR)
        else:  # ケース 4
            if img.shape[0] != norm_img_size:
                _size = (norm_img_size, norm_img_size)
                img = center_crop(img, _size)
                # random_cropからコールしているfixed_cropを参照から代入に変えたので不要に
                img = cv2.resize(img, (norm_img_size, norm_img_size), interpolation=cv2.INTER_LINEAR)

    return img

# 分岐がわかりにくいので整理
def crop_proc_v2(img, do_crop, src_img_type, norm_img_size, rand_crop, crop_ratio):

    def rand_crop_proc(img):
        ##################################################################
        crop_range = abs(1.0 - crop_ratio)

        # 画像からマージンを除外したサイズを算出
        # ★ マージン付き正規化画像の場合、crop ratioが変化したら対応できないことは無視
        size_wo_margin = int(img.shape[0] / (1.0 + crop_range))
        if size_wo_margin % 2 == 1:
            size_wo_margin = size_wo_margin + 1

        # cropするサイズを固定にする実装
        # _size = (size_wo_margin, size_wo_margin)

        # cropするサイズを可変にする実装
        crop_range = np.random.rand() * crop_range * 2.0 + 1.0 - crop_range
        crop_size = int(size_wo_margin * crop_range)
        _size = (crop_size, crop_size)
        ##################################################################

        img = random_crop(img, _size)
        return img

    # そのまま使える画像であればcrop処理にまわさない
    if do_crop:

        # 入力画像(=データセットの画像そのもの ≠関数に入力された画像)に応じた処理
        if src_img_type == "original" or src_img_type == "normalized with margin":

            # src_img_type == "original"
            # データセットの画像はオリジナル画像(=非正規化画像)
            # この関数に入る前にcrop率を考慮してマージンをつけて正規化している

            # src_img_type == "normalized with margin"
            # データセットの画像は正規化してあり、かつマージンが付与されている

            if rand_crop == 1:  # random cropする
                img = rand_crop_proc(img)
            else:
                # 学習時にcenter cropされることは基本ない
                # 評価時はそもそもcropされることは基本ない
                # → 基本このパスには入らない

                # 画像からマージンを除外したサイズを算出
                # ★ マージン付き正規化画像の場合、crop ratioが変化したら対応できないことは無視
                crop_range = abs(1.0 - crop_ratio)
                size_wo_margin = int(img.shape[0] / (1.0 + crop_range))
                if size_wo_margin % 2 == 1:
                    size_wo_margin = size_wo_margin + 1

                _size = (size_wo_margin, size_wo_margin)

                img = center_crop(img, _size)

        elif src_img_type == "normalized without margin":
            # src_img_type == "normalized without margin"
            # データセットの画像は正規化してあるが、マージンが付与されていない

            if rand_crop == 1:  # random cropする
                # マージンを付けたサイズを算出
                margin_ratio = 1.0 - crop_ratio
                size_w_margin = int((1.0 + margin_ratio) * img.shape[0])
                if size_w_margin % 2 == 1:
                    size_w_margin = size_w_margin + 1

                # 0(or 127)埋めした画像に張り付けて、マージン付与画像にする
                _size = (size_w_margin, size_w_margin, img.shape[2])
                if img.dtype == np.uint8:
                    marginalized_img = np.ones(_size, dtype=img.dtype)
                    marginalized_img *= 127
                elif img.dtype == np.float32:
                    marginalized_img = np.zeros(_size, dtype=img.dtype)
                start = (size_w_margin - img.shape[0]) // 2
                end = start + img.shape[0]
                marginalized_img[start:end, start:end, ] = img

                # crop処理の実体
                img = rand_crop_proc(marginalized_img)
            # else:
            #     # マージンなしで正規化されているので何もしない

    # CNNの入力画像サイズにあわせる
    h, _, _ = img.shape
    if h != norm_img_size:
        img = cv2.resize(img, (norm_img_size, norm_img_size), interpolation=cv2.INTER_LINEAR)

    return img

def cutoff_proc(img, cutoff):
    """
    LIUJIN:
        Japan version does it on pixel value range of (-1, 1).
        Modify to on range of (0, 255).
    """
    _rd = random.randint(0, cutoff)
    if _rd == 0:
        # set random position
        centerh = random.randint(0, img.shape[0] - 1)
        centerw = random.randint(0, img.shape[1] - 1)

        # random
        while True:
            half_w = int(random.randint(2, int((img.shape[0] - 1))) / 2)
            aspect_r = np.random.uniform(-np.log(3.0), np.log(3.0))
            aspect_r = np.exp(aspect_r)
            half_h = int(half_w * aspect_r)
            starth = max(0, centerh - half_h)
            endh = min(img.shape[0], centerh + half_h)
            startw = max(0, centerw - half_w)
            endw = min(img.shape[1], centerw + half_w)
            tmp_size = float((endh - starth) * (endw - startw))
            tmp_size = tmp_size / float(img.shape[0] * img.shape[1])
            if tmp_size >= 0.02 and tmp_size <= 0.4:
                break
        # rand_fp = random.uniform(-1.0, 1.0)
        rand_fp = random.randint(0, 255)
        img[starth:endh, startw:endw, :] = rand_fp

    return img