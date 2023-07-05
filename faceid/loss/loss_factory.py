import torch.nn as nn

from .multidata_v1 import MultidataLoss, MultidataPartialFC
from .face_transformer import CosFace
from .insightface import ArcFace


def build_head(name: str, embed_size: int, num_classes: int, **kwargs) -> nn.Module:
    if name == 'cosface':
        head = CosFace(embed_size, num_classes)
    elif name =='arcface':
        head = ArcFace(embed_size, num_classes)
    else:
        print('Unknown loss name, abandoning training!')
        return

    return head

def build_multidata_head(name: str, embed_size: int, num_classes: list, wsize: int,
                         partial_fc: bool ,**kwargs) -> nn.Module:
    if not partial_fc:
        head = MultidataLoss(name, embed_size, num_classes, **kwargs)
    else:
        head = MultidataPartialFC(name, embed_size, num_classes, wsize, **kwargs)

    return head