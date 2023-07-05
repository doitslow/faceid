import random

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import Sampler, BatchSampler


def index_dataset(dataset):
    cls_to_ind = {}
    # for idx, lbl in enumerate(dataset.labels):
    for idx, lbl in enumerate(dataset.lbls):
        if lbl in cls_to_ind:
            cls_to_ind[lbl].append(idx)
        else:
            cls_to_ind[lbl] = [idx]
    return cls_to_ind


#=========#=========#=========#=========#=========#=========#=========#=========
class Cuser(Sampler):
    def __init__(self, data_source: ImageFolder, batch_size, class_per_batch=64,
                 iter_per_epoch=200):
        super(Sampler, self).__init__()
        self.cpb = class_per_batch
        self.batch_size = batch_size
        self.n_batch = iter_per_epoch
        self.class_idx = list(range(data_source.num_class))
        # self.class_idx = data_source.uni_classes

        # self.class_idx = []
        # for i in data_source.labels:
        #     if i not in self.class_idx:
        #         self.class_idx.append(i)

        self.images_by_class = index_dataset(data_source)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            selected_class = random.sample(self.class_idx, k=self.cpb)
            example_indices = []

            for c in selected_class:
                if c not in self.images_by_class:
                    # print(str(c))
                    continue
                else:
                    img_ind_of_cls = self.images_by_class[c]
                    new_ind = random.sample(img_ind_of_cls,
                                            k=min(len(img_ind_of_cls), int(self.batch_size/self.cpb)))
                    example_indices += new_ind

            while len(example_indices) < self.batch_size:
                [extra_class] = random.sample(self.class_idx, k=1)
                if extra_class not in self.images_by_class:
                    # print(str(extra_class))
                    continue
                else:
                    img_ind_of_cls = self.images_by_class[extra_class]
                    new_ind = random.sample(img_ind_of_cls,
                                            k=min(len(img_ind_of_cls), int(self.batch_size/self.cpb)))
                    example_indices += new_ind

            yield example_indices[:self.batch_size]


class DistributedSampler(Sampler):
    """ Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
            if not torch.distributed.is_initialized():
                raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                torch.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = torch.distributed.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

    def __iter__(self):
        return iter(
            [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

    def __len__(self):
        return len(self.iterable)


class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.util.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.util.data.sampler import BatchSampler
        >>> from torch.util.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)
