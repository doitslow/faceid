import os
import logging
from abc import ABC, abstractmethod
from easydict import EasyDict as Edict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from util.decorate import rank0_only, starit

"""
    Let learner be the learning status keeper. It does not own any component used for training/evaluating,
    its purpose is to track the status of learning and deals with setting up environment.
"""

class AbstractLearner(ABC):
    """Abstract learner
        - includes handling of distributed training
    """

    def __init__(self, config):
        self.config = Edict()  # A copy of input Namespace in dict format
        for k, v in vars(config).items():
            self.config[k] = v

        self.ddp = config.ddp
        self.job_space = config.job_space

        # ==================== handle distributed training =====================
        self.wsize, self.wrank, self.lrank = None, None, None
        self.rank0, self.map_loc = None, None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.batch_size = config.batch_size # for non-distributed training,  use batch size
        self.loader_batch = config.batch_size # for non-distributed training, loader batch size is total batch size
        if self.ddp is not None:
            self.wsize, self.wrank, self.lrank = self.init_distributed()
            self.rank0 = self.wrank == 0
            self.device = torch.device("cuda", self.lrank)
            self.batch_size = config.batch_size * self.wsize
            self.map_loc = {'cuda:0': 'cuda:{}'.format(self.lrank)}

    def init_distributed(self):
        wsize = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        wrank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        lrank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        os.environ['MASTER_PORT'] = '20000'
        dist.init_process_group('nccl', init_method='file://' + self.job_space + '/shared',
                                world_size=wsize, rank=wrank)

        return wsize, wrank, lrank

    def model_parallel(self, model, find_unused=True):
        if self.ddp == 'torch':
            wrapped_model = NativeDDP(model, device_ids=[self.lrank], find_unused_parameters=find_unused)
        else:
            print("Unknown distributed training method!")

        return wrapped_model

    @rank0_only  # if distributed, only do action if rank==0
    def log(self, msg):
        logging.info(msg)

    @starit
    @rank0_only
    def star_log(self, msg):
        logging.info(msg)

    @rank0_only
    def log_args(self, param):
        self.star_log('Learner called with parameters as follow')
        for k, v in sorted(param.items()):
            logging.info("{}: {}".format(k, v))
        self.star_log('End of parameters')

    @abstractmethod
    def load_ckpt(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_ckpt(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_train_loader(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_eval_loader(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_model(self, *args, **kwargs):
        """
            In order to cope with TIMM or other main stream format, use model wrapper to wrap
            backbone + head into model. In this way, it is much easier to use third party optimizer
            without hiccups.
        """
        pass

    @abstractmethod
    def get_criterion(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, *args, **kwargs):
        pass




