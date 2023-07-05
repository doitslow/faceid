import sys
import logging
import torch
import torch.nn as nn

def load_backbone(backbone, loaded_ckpt, strict_load): # Fixme: a general logger that only logs on device: 0?
    key = 'backbone'
    if 'model' in loaded_ckpt.keys():
        key = 'model'
    elif 'net' in loaded_ckpt.keys():
        key = 'net'
    elif 'state_dict' in loaded_ckpt.keys():
        key = 'state_dict'
    # print("The key for model/backbone in loaded state dict is '{}'".format(key))

    try:
        backbone.load_state_dict(loaded_ckpt[key], strict=strict_load)
    except RuntimeError:
        try:
            backbone_state_dict = {k.replace('backbone.', ''): v for k, v in loaded_ckpt[key].items() if 'backbone' in k}
            backbone.load_state_dict(backbone_state_dict, strict=strict_load)
        except RuntimeError:
            logging.info("Cannot match up keys for loading state dict! ")

    return backbone

def build_backbone(name: str, **kwargs):
    if name == 'swin_s_p2w7_112':
        from .swin_transformer import swin_s_p2w7_112
        backbone = swin_s_p2w7_112(**kwargs)
        embed_size = backbone.num_features  # embedding size is determined in backbone definition
        logging.info("Backbone {} created!".format(name))
    elif name == 'resnet50':
        from .resnet import resnet50
        backbone = resnet50(**kwargs)
        embed_size = backbone.embed_size
    else:
        raise ValueError('Not implemented!')
        # sys.exit("Unknown network!")

    return backbone, embed_size


class ModelWrapper(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
        embed_size: embedding size of face model which is critical when used for evaluation
        return_embed: whether or not to return embeddings during training
    """

    def __init__(self, backbone, head, return_embed=False):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.head = head
        self.return_embed = return_embed
        self.embed_size = None  # make embedding size an attribute

    def forward(self, data, label=None):
        embed = self.backbone.forward(data)
        if label is None:
            return embed
        else:
            pred = self.head.forward(embed, label)
            if self.return_embed:
                return pred, embed
            else:
                return pred


class MultidataModelWrapper(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, backbone, head, return_embed=False):
        """
        """
        super(MultidataModelWrapper, self).__init__()
        self.backbone = backbone
        self.head = head
        self.return_embed = return_embed
        self.embed_size = None  # make embedding size an attribute

    def forward(self, data, label=None, dataset_id=None, wsize=None, wrank=None):
        embed = self.backbone.forward(data)
        if label is None:
            return embed
        else:
            pred = self.head.forward(embed, label, dataset_id, wsize, wrank)
            if self.return_embed:
                return pred, embed
            else:
                return pred