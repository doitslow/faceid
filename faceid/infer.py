import os
import logging
from datetime import datetime
from os.path import basename, join, dirname, exists, abspath
from shutil import copytree, rmtree, ignore_patterns, copy2

from train import get_learner
from learner import BaseLearner
from temp.r100 import create_r100, get_model

r100_config = {
    "act_func": "prelu",
    "attn_meth": None,
    "bn_eps": 2e-05,
    "bn_mom": 0.1,
    "conv_type": "vanilla",
    "dropblock_setting": [0.1, 7],
    "dropout": 0.4,
    "embd_size": 256,
    "feat_list": "1,2,3,4",
    "net_build": "fresnet",
    "net_in": "1",
    "net_out": "E",
    "use_dropblock": "N",
    "use_ibn": "Y",
    "num_layer": 100,
    "net_width": [
                    64,
                    64,
                    128,
                    256,
                    512
                ],
}


def infer_r100():
    learner = get_learner(learner_class=BaseLearner, training=False)
    # model = create_r100()
    model = get_model(**r100_config)
    # print("Are we here?")
    learner.infer(model=model)


#=========#=========#=========#=========#=========#=========#=========#=========
def main():
    learner = get_learner()
    job_space = dirname(learner.config.resume_ckpt)
    model_pths = [join(job_space, f) for f in os.listdir(job_space) if f.endswith('.pth.tar')]
    for model_pth in model_pths:
        print("Inferencing {}".format(model_pth))
        learner.infer(model_pth)


if __name__ == '__main__':
    # main()
    infer_r100()
