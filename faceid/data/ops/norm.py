# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:35:29 2016

@author: yasu
"""

import numpy as np
import cv2
import math

def part_face_crop_align(img, pts, dst_size, PartNo, margin_ratio=0.0):
    # RetinaFace 5point用
    eye1 = tuple(np.array([pts[0], pts[1]]))
    eye2 = tuple(np.array([pts[2], pts[3]]))
    mouth = tuple(np.array([(pts[6] + pts[8]) / 2, (pts[7] + pts[9]) / 2]))

    Dt = np.array([[eye1[0], eye1[1], 1, 0],
                   [eye2[0], eye2[1], 1, 0],
                   [mouth[0], mouth[1], 1, 0],
                   [eye1[1], -eye1[0], 0, 1],
                   [eye2[1], -eye2[0], 0, 1],
                   [mouth[1], -mouth[0], 0, 1]])

    # 正規化画像の出力が256x256を基準に正規化座標が設定されているため、
    # 基準からのスケールを算出
    scale = 256.0 / dst_size

    Ds = np.array([[397.0], [544.0], [470.0], [405.0], [405.0], [552.0]])
    # Ds = Ds / 4.0
    # 正規化座標をスケールに合わせて変換
    Ds = Ds / scale

    TR = Dt.T.dot(Dt)
    ITR = np.linalg.inv(TR)
    TITR = ITR.dot(Dt.T)
    warp_M = TITR.dot(Ds)

    warp_mat = np.array([[warp_M[0], warp_M[1], warp_M[2]], [-warp_M[1], warp_M[0], warp_M[3]]])

    FS = np.ones((pts.shape[0] // 2, 3))
    for i in range(pts.shape[0] // 2):
        FS[i][0] = pts[i * 2]
        FS[i][1] = pts[i * 2 + 1]
    FDT = np.ndarray((pts.shape[0] // 2, 2), dtype=np.float32)
    # アフィン変換後の顔特徴点座標を算出
    for (i, data) in enumerate(FS):
        FDT[i] = data.dot(warp_mat.T)

    output_size = math.ceil((1 + margin_ratio) * dst_size)

    # 画像出力時に口中心が画像中心となるようなオフセット算出
    if PartNo == 2:
        CropCenterPoint_offset = (output_size / 2.0) - ((FDT[0] + FDT[1]) / 2.0)
    elif PartNo == 3:
        CropCenterPoint_offset = (output_size / 2.0) - ((FDT[3] + FDT[4]) / 2.0)

    # アフィン変換行列にオフセット加算
    warp_mat[0][2] += CropCenterPoint_offset[0]
    warp_mat[1][2] += CropCenterPoint_offset[1]

    dst = cv2.warpAffine(img, warp_mat, (output_size, output_size))

    return dst, warp_mat

def under_face_crop_align(img, pts, dst_size, margin_ratio=0.0):

    # RetinaFace 5point用
    eye1 = tuple(np.array([pts[0], pts[1]]))
    eye2 = tuple(np.array([pts[2], pts[3]]))
    mouth = tuple(np.array([(pts[6] + pts[8]) / 2, (pts[7] + pts[9]) / 2]))
    
    Dt = np.array([[eye1[0], eye1[1], 1, 0],
                   [eye2[0], eye2[1], 1, 0],
                   [mouth[0], mouth[1], 1, 0],
                   [eye1[1], -eye1[0], 0, 1],
                   [eye2[1], -eye2[0], 0, 1],
                   [mouth[1], -mouth[0], 0, 1]])    

    # 正規化画像の出力が256x256を基準に正規化座標が設定されているため、
    # 基準からのスケールを算出
    scale = 256.0 / dst_size
    
    Ds = np.array([[397.0], [544.0], [470.0], [405.0], [405.0], [552.0]])
    #Ds = Ds / 4.0
    # 正規化座標をスケールに合わせて変換
    Ds = Ds / scale
    
    TR = Dt.T.dot(Dt)
    ITR = np.linalg.inv(TR)
    TITR = ITR.dot(Dt.T)
    warp_M = TITR.dot(Ds)
    
    warp_mat = np.array([[warp_M[0], warp_M[1], warp_M[2]], [-warp_M[1], warp_M[0], warp_M[3]]])
    
    FS = np.ones((pts.shape[0]//2, 3))
    for i in range(pts.shape[0]//2):
        FS[i][0] = pts[i * 2]
        FS[i][1] = pts[i * 2 + 1]
    FDT = np.ndarray((pts.shape[0]//2, 2), dtype=np.float32)
    # アフィン変換後の顔特徴点座標を算出
    for (i, data) in enumerate(FS):
        FDT[i] = data.dot(warp_mat.T)
    
    output_size = math.ceil((1 + margin_ratio) * dst_size)

    # 画像出力時に口中心が画像中心となるようなオフセット算出
    CropCenterPoint_offset = (output_size / 2.0) - ((FDT[3] + FDT[4]) / 2.0)

    # アフィン変換行列にオフセット加算
    warp_mat[0][2] += CropCenterPoint_offset[0]
    warp_mat[1][2] += CropCenterPoint_offset[1]

    dst = cv2.warpAffine(img, warp_mat, (output_size, output_size))

    return dst, warp_mat

    
def crop_align(img, pts):

    # RetinaFace 5point用
    eye1 = tuple(np.array([pts[0], pts[1]]))
    eye2 = tuple(np.array([pts[2], pts[3]]))
    mouth = tuple(np.array([(pts[6] + pts[8]) / 2, (pts[7] + pts[9]) / 2]))
    
    Dt = np.array([[eye1[0], eye1[1], 1, 0],
                   [eye2[0], eye2[1], 1, 0],
                   [mouth[0], mouth[1], 1, 0],
                   [eye1[1], -eye1[0], 0, 1],
                   [eye2[1], -eye2[0], 0, 1],
                   [mouth[1], -mouth[0], 0, 1]])    
    
    Ds = np.array([[397.0], [544.0], [470.0], [405.0], [405.0], [552.0]])
    Ds = Ds / 4.0
    
    TR = Dt.T.dot(Dt)
    ITR = np.linalg.inv(TR)
    TITR = ITR.dot(Dt.T)
    warp_M = TITR.dot(Ds)
    
    warp_mat = np.array([[warp_M[0], warp_M[1], warp_M[2]], [-warp_M[1], warp_M[0], warp_M[3]]])
    
    #dst = cv2.warpAffine(img, warp_mat, (1000, 1000))
    dst = cv2.warpAffine(img, warp_mat, (250, 250))

    FS = np.ones((pts.shape[0]//2, 3))
    for i in range(pts.shape[0]//2):
        FS[i][0] = pts[i * 2]
        FS[i][1] = pts[i * 2 + 1]
    FDT = np.ndarray((pts.shape[0]//2, 2), dtype=np.float32)
    for (i, data) in enumerate(FS):
        FDT[i] = data.dot(warp_mat.T)

    return dst, FDT, warp_mat

def crop_align_ori1(img, pts, margin_ratio=0.0):
    norm_point = np.array([[397.0, 305.0], [493.0, 305.0], [445.0, 401.0]])
    norm_point = norm_point / 4.0

    # RetinaFace 5point用
    eye1 = tuple(np.array([pts[0], pts[1]]))
    eye2 = tuple(np.array([pts[2], pts[3]]))
    mouth = tuple(np.array([(pts[6]+pts[8])/2, (pts[7]+pts[9])/2]))

    Dt = np.array([[eye1[0], eye1[1], 1, 0],
                   [eye2[0], eye2[1], 1, 0],
                   [mouth[0], mouth[1], 1, 0],
                   [eye1[1], -eye1[0], 0, 1],
                   [eye2[1], -eye2[0], 0, 1],
                   [mouth[1], -mouth[0], 0, 1]])

    Ds = np.array([[397.0], [493.0], [445.0], [305.0], [305.0], [401.0]])
    Ds = Ds / 4.0
    
    TR = Dt.T.dot(Dt)
    ITR = np.linalg.inv(TR)
    TITR = ITR.dot(Dt.T)
    warp_M = TITR.dot(Ds)
    
    warp_mat = np.array([[warp_M[0], warp_M[1], warp_M[2]], [-warp_M[1], warp_M[0], warp_M[3]]])

    # アフィン変換
    #dst = cv2.warpAffine(img, warp_mat, (1000, 1000))
    dst = cv2.warpAffine(img, warp_mat, (250, 250))

    # 対象領域の切り出し
    #offset = 80.0
    offset = 20.0
    xlow = int(norm_point[0][0] - offset)
    xhigh = int(norm_point[1][0] + offset)
    ylow = int(norm_point[0][1] - offset)
    yhigh = int(norm_point[2][1] + offset)

    # マージンの設定
    if margin_ratio != 0.0:
        tmp_w = xhigh- xlow
        tmp_h = yhigh- ylow
        tmp_w_ = (int)((1.0 + margin_ratio) * tmp_w)
        tmp_h_ = (int)((1.0 + margin_ratio) * tmp_h)
        margin_w = (tmp_w_ - tmp_w) // 2
        margin_h = (tmp_h_ - tmp_h) // 2
        offset_w = offset + margin_w
        offset_h = offset + margin_h
        xlow = int(norm_point[0][0] - offset_w)
        xhigh = int(norm_point[1][0] + offset_w)
        ylow = int(norm_point[0][1] - offset_h)
        yhigh = int(norm_point[2][1] + offset_h)

    patch = dst[ylow:yhigh, xlow:xhigh, ]
    
    return patch, warp_mat

def calcRange(mdPts, PartNo, margin_ratio=0.0):

    #PATCH_SIZE = 256
    PATCH_SIZE = 64
    HALF_PATCH_SIZE = PATCH_SIZE // 2

    #ptsCenters = np.ndarray((2, 2), dtype=np.int32)
    ptsCenters = np.ndarray((2, 2))
    ptsCenters[0] = (mdPts[0] + mdPts[1]) / 2.0 + 0.5
    ptsCenters[1] = (mdPts[3] + mdPts[4]) / 2.0 + 0.5
    ptsCenters = ptsCenters.astype(np.int32)

    LT = ptsCenters[PartNo - 2] - HALF_PATCH_SIZE
    RB = ptsCenters[PartNo - 2] + HALF_PATCH_SIZE

    # マージンの設定
    if margin_ratio != 0.0:
        xlow = LT[0]
        xhigh = RB[0]
        ylow = LT[1]
        yhigh = RB[1]
        tmp_w = xhigh - xlow
        tmp_h = yhigh - ylow
        tmp_w_ = (int)((1.0 + margin_ratio) * tmp_w)
        tmp_h_ = (int)((1.0 + margin_ratio) * tmp_h)
        margin_w = (tmp_w_ - tmp_w) // 2
        margin_h = (tmp_h_ - tmp_h) // 2
        LT[0] = LT[0] - margin_w
        RB[0] = RB[0] + margin_w
        LT[1] = LT[1] - margin_h
        RB[1] = RB[1] +  margin_h

    return (LT[0], RB[0], LT[1], RB[1])

def generate_patch(img, FA_data, PartNo, margin_ratio=0.0):
        
    if PartNo == 1:
        dst, warp_mat = crop_align_ori1(img, FA_data, margin_ratio)
    elif PartNo == 2:
        # back up
        # dst, FDT, warp_mat = crop_align(img, FA_data)
        # CropRange = calcRange(FDT, PartNo, margin_ratio)
        # dst = dst[CropRange[2]:CropRange[3], CropRange[0]:CropRange[1], ]
        dst, warp_mat = part_face_crop_align(img, FA_data, 112, PartNo, margin_ratio)
    elif PartNo == 3:
        dst, warp_mat = under_face_crop_align(img, FA_data, 112, margin_ratio)

    return dst, warp_mat

def generate_patch_fd_crop(img, rect, margin_ratio=0.0):
    center = [(rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2]
    length = max(rect[2] - rect[0], rect[3] - rect[1])
    RoI = [center[0] - length // 2,
           center[1] - length // 2,
           center[0] + length // 2,
           center[1] + length // 2]
    length = (length // 2) * 2

    if margin_ratio != 0.0:
        tmp_length = int((1.0 + margin_ratio) * length)
        margin = (tmp_length - length) // 2
        RoI[0] = RoI[0] - margin
        RoI[1] = RoI[1] - margin
        RoI[2] = RoI[2] + margin
        RoI[3] = RoI[3] + margin
        tmp_length = RoI[2] - RoI[0]
    else:
        tmp_length = length
        margin = 0

    # 最初に大きい画像を作っておき，領域外アクセスを回避
    # 計算が遅くなるので変える
    # long_side = max(img.shape[0], img.shape[1])
    # imgLarge = np.ones((long_side * 2, long_side * 2, 3)).astype(np.uint8) * 128
    # imgLarge[long_side // 2: long_side // 2 + img.shape[0],
    #          long_side // 2: long_side // 2 + img.shape[1], :] = img
    #
    # FDcropImg = imgLarge[RoI[1] + long_side // 2: RoI[3] + long_side // 2,
    #                      RoI[0] + long_side // 2: RoI[2] + long_side // 2]

    # 切り出し範囲が画像外になるかどうか事前チェックし，
    # 参照可能な範囲だけコピーする
    FDcropImg = np.ones((tmp_length, tmp_length, 3)).astype(np.uint8) * 128
    dst_ROI = [0, 0, tmp_length, tmp_length]

    # print("length", length)
    # print("tmp_length", tmp_length)
    # print("img.shape", img.shape)
    # print("RoI", RoI)
    # print("dst_ROI", dst_ROI)

    if RoI[0] < 0:
        dst_ROI[0] -= RoI[0]
        RoI[0] = 0
    if RoI[1] < 0:
        dst_ROI[1] -= RoI[1]
        RoI[1] = 0
    if RoI[2] > img.shape[1]:
        dst_ROI[2] -= (RoI[2] - img.shape[1])
        RoI[2] = img.shape[1]
    if RoI[3] > img.shape[0]:
        dst_ROI[3] -= (RoI[3] - img.shape[0])
        RoI[3] = img.shape[0]

    FDcropImg[dst_ROI[1]: dst_ROI[3], dst_ROI[0]: dst_ROI[2]] = img[RoI[1]: RoI[3], RoI[0]: RoI[2]]

    return FDcropImg

    # make_cropped_eval_dataを動かすときはこちら
    #return FDcropImg, RoI, dst_ROI