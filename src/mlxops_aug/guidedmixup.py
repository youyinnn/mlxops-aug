from dataclasses import dataclass
from types import SimpleNamespace
from .aug_base import *

try:
    from pairing import onecycle_cover  # type: ignore
except:
    print()
    print("*******")
    print(
        "Pairing module for onecycle_cover is not available. Can only use the random condition, no greedy condition."
    )
    print("*******")
    print()
    pass

import torchvision.transforms.functional as TF
from mlxops_xai import gradient


@dataclass
class GuidedMixup(AugmentBase):

    def __post_init__(self):
        args = SimpleNamespace()

        default_args = [
            ("condition", "random"),
            ("block_num", None),
        ]

        for da in default_args:
            arg_key, default_v = da
            setattr(args, arg_key, self.config.get(arg_key, default_v))

        self.args = args

    def aug(self, _x, _y, model):
        # y is not
        sa = gradient.vanilla_gradient(model, _x, _y.argmax(1))
        return mixup_process(_x, _y, grad=sa, args=self.args)

    def __call__(self, _x, _y, model) -> AugResult:
        if torch.rand(1) <= self.config.get("prob", 1.0):
            _y_oh = torch.nn.functional.one_hot(
                _y, num_classes=self.num_classes)
            _x, _y = self.aug(_x, _y_oh, model)

        return AugResult(
            augmented_x=_x,
            augmented_y=_y
        )


def cosine_similarity(a, b):
    dot = a.matmul(b.t())
    norm = a.norm(dim=1, keepdim=True).matmul(b.norm(dim=1, keepdim=True).t())
    return dot / norm


def distance_function(a, b=None, distance_metric="jsd"):
    """pytorch distance
    input:
     - a: (batch_size1 N, n_features)
     - b: (batch_size2 M, n_features)
    output: NxM matrix"""
    if b is None:
        b = a
    if distance_metric == "cosine":
        distance = 1 - \
            cosine_similarity(a.view(a.shape[0], -1), b.view(b.shape[0], -1))
    elif distance_metric == "cosine_abs":
        distance = (
            1 -
            cosine_similarity(
                a.view(a.shape[0], -1), b.view(b.shape[0], -1)).abs()
        )
    elif distance_metric == "l1":
        ra = a.view(a.shape[0], -1).unsqueeze(1)
        rb = b.view(b.shape[0], -1).unsqueeze(0)
        distance = (ra - rb).abs().sum(dim=-1).view(a.shape[0], b.shape[0])
    elif distance_metric == "l2":
        ra = a.view(a.shape[0], -1).unsqueeze(1)
        rb = b.view(b.shape[0], -1).unsqueeze(0)
        distance = ((ra - rb).norm(dim=-1)).view(a.shape[0], b.shape[0])
    else:
        raise NotImplementedError
    return distance


def pairing(sc_a, sc_b=None, condition="random", distance_metric="l2"):
    """
    - skipping -
    if idx > images.size(0)*0.8:
        sorted_indices[row]=row
        # non_skip_idx.append(row) # if you don't select, then delete it
    else:
        sorted_indices[row]=col
        # non_skip_idx.append(row)
    """
    if sc_b is None:
        sc_b = sc_a

    if condition.startswith("greedy"):
        X = distance_function(sc_a, sc_b, distance_metric).cpu().numpy()
        sorted_indices = onecycle_cover(X)
    else:  # random
        sorted_indices = np.random.permutation(sc_a.size(0))
    return sorted_indices


def get_lambda(alpha=1.0, alpha2=None):
    """Return lambda"""
    if alpha > 0.0:
        if alpha2 is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(alpha + 1e-2, alpha2 + 1e-2)
    else:
        lam = 1.0
    return lam


def mixup_process(
    out,
    target_reweighted,
    grad=None,
    args=None,
):
    """
    Credit: https://github.com/3neutronstar/GuidedMixup/blob/main/mixup.py
    """

    condition = args.condition

    ratio_b = None
    indices = np.random.permutation(out.size(0))

    sc_a = grad
    # guidedmixup
    sc_a = TF.gaussian_blur(sc_a, (7, 7), (3, 3))
    sc_a /= (sc_a).sum(dim=[-1, -2], keepdim=True)
    if condition.startswith("greedy"):
        try:
            indices = pairing(
                sc_a.detach(), condition="greedy_c", distance_metric="l2")
        except:
            indices = pairing(
                sc_a.detach(), condition=condition, distance_metric="l2")
    else:
        indices = pairing(sc_a.detach(), condition=condition,
                          distance_metric="l2")

    out_b = out[indices]
    sc_b = sc_a[indices]
    norm_sc_a = torch.div(sc_a, (sc_a + sc_b).detach())
    ratio = norm_sc_a.mean(dim=[-1, -2]).unsqueeze(-1)
    mask_a = torch.stack([norm_sc_a] * 3, dim=1)
    out = mask_a * out + (1 - mask_a) * out_b
    ratio_b = 1.0 - ratio

    target_shuffled_onehot = target_reweighted[indices]
    if ratio.dim() == 1:
        ratio = ratio.unsqueeze(-1)
    if ratio_b is None:
        target_reweighted = target_reweighted * ratio + target_shuffled_onehot * (
            1 - ratio
        )
    else:
        target_reweighted = target_reweighted * ratio + target_shuffled_onehot * (
            ratio_b
        )
    # save "out" in image format
    # torchvision.utils.save_image(out, 'out.png',normalize=True)
    # print("SAVE")
    # print(time.time()-tik)
    return out, target_reweighted
