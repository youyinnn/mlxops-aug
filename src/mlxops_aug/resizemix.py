from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import interpolate

from .aug_base import *

# Implementation Credit: https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/augments/resizemix.py


@dataclass
class ResizeMix(AugmentBase):

    def __call__(self, _x, _y) -> AugResult:
        lam = 1
        if torch.rand(1) <= self.config.get("prob", 1.0):
            _x, (ya, yb, lam) = resizemix(_x, _y, **self.config)
            _y = self.get_mixed_y_from_ablam(ya, yb, lam)

        return AugResult(
            augmented_x=_x,
            augmented_y=_y,
            lam=lam
        )


@torch.no_grad()
def resizemix(
    img,
    gt_label,
    scope=(0.1, 0.8),
    dist_mode=False,
    alpha=1.0,
    lam=None,
    use_alpha=False,
    interpolate_mode="nearest",
    return_mask=False,
    **kwargs
):
    r"""ResizeMix augmentation.

    "ResizeMix: Mixing Data with Preserved Object Information and True Labels
    (https://arxiv.org/abs/2012.11101)".

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        use_alpha (bool): Whether to use alpha instead of scope. Notice
            that ResizeMix is designed for supervised learning, it uses
            Uniform discribution rather than Beta. But in SSL contrastive
            learning, it's better to use large alpha.
        interpolate_mode (str): Mode of resizing interpolation {'nearest',
            'bilinear'}. Defaults to 'nearest'.
        scope (float): Sample Uniform distribution to get tao.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    def rand_bbox_tao(size, tao, return_mask=False, device="cpu"):
        """generate random box by tao (scale)"""
        W = size[2]
        H = size[3]
        cut_w = int(W * tao)
        cut_h = int(H * tao)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        if not return_mask:
            return bbx1, bby1, bbx2, bby2
        else:
            mask = torch.zeros((1, 1, W, H)).to(device)
            mask[:, :, bbx1:bbx2, bby1:bby2] = 1
            mask = mask.expand(size[0], 1, W, H)  # (N, 1, H, W)
            return bbx1, bby1, bbx2, bby2, mask

    assert len(scope) == 2

    # normal mixup process
    if not dist_mode:
        rand_index = torch.randperm(img.size(0))
        if len(img.size()) == 4:  # [N, C, H, W]
            img_resize = img.clone()
            img_resize = img_resize[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # * notice that the rank of two groups of img is fixed
            img_resize = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        _, _, h, w = img.size()
        shuffled_gt = gt_label[rand_index]

        # generate tao
        if lam is None:
            if use_alpha == True:
                tao = np.random.beta(alpha, alpha)
                if tao < scope[0] or tao > scope[1]:
                    tao = np.random.uniform(scope[0], scope[1])
            else:
                # original settings in ResizeMix
                tao = np.random.uniform(scope[0], scope[1])
        else:
            tao = min(max(lam, scope[0]), scope[1])

        # random box
        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox_tao(
                img.size(), tao, device=img.device)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox_tao(
                img.size(), tao, True, device=img.device)
        img_resize = interpolate(
            img_resize, (int(bby2 - bby1), int(bbx2 - bbx1)), mode=interpolate_mode
        )

        img[:, :, bby1:bby2, bbx1:bbx2] = img_resize
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            img = (img, mask)

        return img, (gt_label, shuffled_gt, lam)

    # dist mixup with cross gpus shuffle
    else:
        if len(img.size()) == 5:  # self-supervised img [N, 2, C, H, W]
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True
            )
        else:
            assert len(img.size()) == 4  # normal img [N, C, H, w]
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True
            )
        _, _, h, w = img.size()

        # generate tao
        if lam is None:
            if use_alpha == True:
                tao = np.random.beta(alpha, alpha)
                if tao < scope[0] or tao > scope[1]:
                    tao = np.random.uniform(scope[0], scope[1])
            else:
                # original settings in ResizeMix
                tao = np.random.uniform(scope[0], scope[1])
        else:
            tao = lam

        # random box
        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox_tao(
                img.size(), tao, device=img.device)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox_tao(
                img.size(), tao, True, device=img.device)
        img_ = interpolate(
            img_, (int(bby2 - bby1), int(bbx2 - bbx1)), mode=interpolate_mode)

        img[:, :, bby1:bby2, bbx1:bbx2] = img_
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            img = (img, mask)

        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True
            )
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)
