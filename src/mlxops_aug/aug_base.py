from typing import Callable

import torch
import random
import time
from torchvision.transforms import v2
from dataclasses import dataclass
import numpy as np


@dataclass(kw_only=True)
class AugmentBase:

    num_classes: int
    config: dict

    def get_mixed_y_from_ablam(self, y_a, y_b, lam):
        batch_size = y_a.shape[0]
        if len(lam.shape) == 0:
            lam = torch.tensor(
                [lam] * batch_size, dtype=torch.float32, device=y_a.device
            )

        oh_a = (
            torch.nn.functional.one_hot(y_a, self.num_classes)
            if len(y_a.shape) == 1
            else y_a
        )
        oh_b = (
            torch.nn.functional.one_hot(y_b, self.num_classes)
            if len(y_b.shape) == 1
            else y_b
        )
        ll = lam.reshape(1, batch_size).repeat(
            self.num_classes, 1).permute(1, 0)
        return (oh_a * ll) + (oh_b * (1 - ll))

    def setup(self, setup_args):
        print(f"No setup implementation for: {type(self).__qualname__}")

    def get_x_y(self, aug_result) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_loss(self, output, aug_result, loss_fn):
        _x, _y = aug_result
        return loss_fn(output, _y)

    def __call__(self, x, y):
        raise NotImplementedError()


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning: torch.distributed.all_gather has no gradient. ***
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_ddp(x, idx_shuffle=None, no_repeat=False):
    """Batch shuffle (no grad), for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        return: x, idx_shuffle, idx_unshuffle.
        *** no repeat (09.23 update) ***

    Args:
        idx_shuffle: Given shuffle index if not None.
        no_repeat: The idx_shuffle does not have any repeat index as
            the original indice [i for i in range(N)]. It's used in
            mixup methods (self-supervisedion).
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    if idx_shuffle is None:
        # generate shuffle idx
        idx_shuffle = torch.randperm(batch_size_all).to(x.device)
        # each idx should not be the same as the original
        if bool(no_repeat) == True:
            idx_original = torch.tensor(
                [i for i in range(batch_size_all)]).to(x.device)
            idx_repeat = False
            for i in range(20):  # try 20 times
                if (idx_original == idx_shuffle).any() == True:  # find repeat
                    idx_repeat = True
                    idx_shuffle = torch.randperm(batch_size_all).to(x.device)
                else:
                    idx_repeat = False
                    break
            # repeat hit: prob < 1.8e-4
            if idx_repeat == True:
                fail_to_shuffle = True
                idx_shuffle = idx_original.clone()
                for i in range(3):
                    # way 1: repeat prob < 1.5e-5
                    rand_ = torch.randperm(batch_size_all).to(x.device)
                    idx_parition = rand_ > torch.median(rand_)
                    idx_part_0 = idx_original[idx_parition == True]
                    idx_part_1 = idx_original[idx_parition != True]
                    if idx_part_0.shape[0] == idx_part_1.shape[0]:
                        idx_shuffle[idx_parition == True] = idx_part_1
                        idx_shuffle[idx_parition != True] = idx_part_0
                        if (idx_original == idx_shuffle).any() != True:  # no repeat
                            fail_to_shuffle = False
                            break
                # fail prob -> 0
                if fail_to_shuffle == True:
                    # way 2: repeat prob = 0, but too simple!
                    idx_shift = np.random.randint(1, batch_size_all - 1)
                    idx_shuffle = torch.tensor(  # shift the original idx
                        [
                            (i + idx_shift) % batch_size_all
                            for i in range(batch_size_all)
                        ]
                    ).to(x.device)
    else:
        assert (
            idx_shuffle.size(0) == batch_size_all
        ), "idx_shuffle={}, batchsize={}".format(idx_shuffle.size(0), batch_size_all)

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_shuffle, idx_unshuffle
