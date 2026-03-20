from dataclasses import dataclass

import torch
import numpy as np
from .aug_base import *

# Implementation Credit: https://github.com/ntucllab/imbalanced-DL/blob/main/imbalanceddl/strategy/_remix_drw.py


@dataclass
class ReMix(AugmentBase):

    def __post_init__(self):
        self.cls_num_list = None

    def __call__(self, _x, _y) -> AugResult:
        y_a, y_b, lam_x, lam_y = _y, None, 1.0, 1.0
        if torch.rand(1) <= self.config.get("prob", 1.0):
            if self.cls_num_list is None:
                raise RuntimeError(
                    f"The self.cls_num_list of {type(self).__qualname__} is not defined.")
            if len(_y.shape) > 1:
                no_saliency_aug_idx = torch.where(_y >= 1)[0]
                if no_saliency_aug_idx.shape[0] > 1:
                    _tmp_x = _x[no_saliency_aug_idx]
                    _tmp_y = _y[no_saliency_aug_idx]

                    aug_only_by_resizemix_x, (y_a, y_b, lam_x, lam_y) = remix_data(
                        _tmp_x, _tmp_y, self.cls_num_list, **self.config
                    )

                    aug_only_by_resizemix_y = self.get_mixed_y_from_ablam(
                        y_a, y_b, lam_y)
                    _x[no_saliency_aug_idx] = aug_only_by_resizemix_x
                    _y[no_saliency_aug_idx] = aug_only_by_resizemix_y
            else:
                _x, (y_a, y_b, lam_x, lam_y) = remix_data(
                    _x, _y, self.cls_num_list, **self.config
                )
                _y = self.get_mixed_y_from_ablam(y_a, y_b, lam_y)

        return AugResult(
            augmented_x=_x,
            augmented_y=_y,
            sideproduct=dict(
                y_a=y_a,
                y_b=y_b,
                lam_x=lam_x,
                lam_y=lam_y
            )
        )

    def setup_based_on_datasets(self, train, test):
        all_y = [train[i][1] for i in range(len(train))]
        label_count = {}
        for label in all_y:
            if label_count.get(label) is None:
                label_count[label] = 0
            label_count[label] += 1
        self.cls_num_list = [label_count[i]
                             for i in range(self.num_classes)]

    def get_loss(self, output, aug_result: AugResult, loss_fn):
        if aug_result.sideproduct.get('y_b') is None:
            return super().get_loss(output, aug_result, loss_fn)
        return remix_criterion(
            loss_fn, output,
            aug_result.sideproduct['y_a'],
            aug_result.sideproduct['y_b'],
            aug_result.sideproduct['lam_y']
        )


def remix_data(x, y, cls_num_list, k_majority=3, tau=0.5, alpha=1.0, prob=None):
    """
    Returns mixed inputs, pairs of targets, and lambda_x, lambda_y
    *Args*
    k: hyper parameter of k-majority
    tau: hyper parameter
    where in original paper they suggested to use k = 3, and tau = 0.5
    Here, lambda_y is defined in the original paper of remix, where there
    are three cases of lambda_y as the following:
    (a). lambda_y = 0
    (b). lambda_y = 1
    (c). lambda_y = lambda_x
    """
    if alpha > 0:
        lam_x = np.random.beta(alpha, alpha)
    else:
        lam_x = 1

    # two hyper parameters as Remix suggested, k = 3; \tau = 0.5
    K = k_majority
    cls_num_list = torch.tensor(cls_num_list)

    batch_size = x.size()[0]

    # get the index from random permutation for mix x
    index = torch.randperm(batch_size)

    if len(y.shape) > 1:
        y = y.argmax(dim=1)

    # check list stored pairs of image index where one mixup with the other
    check = []
    for i, j in enumerate(index):
        check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item()])
    check = torch.tensor(check)
    lam_y = []
    for i in range(check.size()[0]):
        # temp1 = n_i; temp2 = n_j
        temp1 = check[i][0]
        temp2 = check[i][1]

        if (temp1 / temp2) >= K and lam_x < tau:
            lam_y.append(0)
        elif (temp1 / temp2) <= (1 / K) and (1 - lam_x) < tau:
            lam_y.append(1)
        else:
            lam_y.append(lam_x)

    lam_y = torch.tensor(lam_y).to(x.device)
    mixed_x = lam_x * x + (1 - lam_x) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, (y_a, y_b, lam_x, lam_y)


def remix_criterion(criterion, pred, y_a, y_b, lam_y):
    """
    In Remix, the lambda for mixing label is different from original mixup.
    """
    # for each y, we calculated its loss individually with their respective
    # lambda_y
    loss = torch.mul(criterion(pred, y_a), lam_y) + torch.mul(
        criterion(pred, y_b), (1 - lam_y)
    )
    return loss.mean()
