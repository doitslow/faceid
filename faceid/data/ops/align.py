import cv2
from skimage import transform as trans
import numpy as np

"""
RetinaFace Alignment
"""
def align(img, landmark=None, src=None, size_in_ArcFaceNorm=100, dst_wh=112, size_w_margin=112):

    dst = np.array(landmark, dtype=np.float32).reshape(5, 2)

    # arcface mean face
    if (not isinstance(src, np.ndarray)):
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
             [33.5493, 92.3655],
             [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0

        # Arc Faceの正規化に対して，どれだけマージンをつけるかを調整 ※ Random Crop用のマージンではない!!
        # デフォルトは100でむしろ外側を取り除く操作を行っている
        # メモ: 行先は112x112の画像
        if size_in_ArcFaceNorm != 112:
            src = src - 56                                      # src 座標系で中心を0に持っていく
            src = src * (112.0 / float(size_in_ArcFaceNorm))    # スケーリングする
            src = src + 56                                      # dst 座標系で中心を戻す

        # 解像度の調整
        if dst_wh != 112:
            src[:] *= float(dst_wh) / 112.0

        # Random Crop用にマージンをつける
        # アフィン変換関数の出力サイズ指定と合わせて調整
        if dst_wh != size_w_margin:
            shift = (size_w_margin - dst_wh) / 2.0
            src[:, 0] += shift
            src[:, 1] += shift

    # アスペクト比を変えない変換行列を推定
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(img, M, (size_w_margin, size_w_margin), borderValue=0.0)
    return warped
    