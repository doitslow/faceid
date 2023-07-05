import os
from torch.autograd import Function
import torch.distributed as dist
import torch


def do_all_gather(tensor, wsize):
    gather_list = [torch.zeros_like(tensor) for _ in range(wsize)]
    dist.barrier()
    torch.distributed.all_gather(gather_list, tensor)
    gathered_tensor = torch.cat(gather_list, dim=0)

    return gathered_tensor


class DistAllGather(Function):

    @staticmethod
    def forward(ctx, input):
        wsize = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        ctx.save_for_backward(input)
        input = input.clone()
        gather_list = [torch.zeros_like(input) for i in range(wsize)]
        dist.barrier()
        dist.all_gather(gather_list, input, dist.group.WORLD)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        dist.barrier()
        dist.reduce_scatter(grad_out, list(grads), group=dist.group.WORLD)
        return grad_out


class AllGather(torch.nn.Module):
    def __init__(self):
        super(AllGather, self).__init__()

    def forward(self, input):
        return DistAllGather.apply(input)


def all_gather(tensor):
    return AllGather()(tensor)

