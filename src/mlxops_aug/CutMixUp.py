from .aug_base import *


@dataclass(kw_only=True)
class CutMixUp(AugmentBase):

    def __post_init__(self):
        cutmix_or_mixup = []
        self.config.setdefault("prob", 1.0)
        self.config.setdefault("use_cutmix", True)
        self.config.setdefault("use_mixup", True)
        self.config.setdefault("alpha", 1.0)

        if self.config["use_cutmix"]:
            cutmix_or_mixup.append(
                v2.CutMix(
                    num_classes=self.num_classes,
                    alpha=self.config["alpha"],
                )
            )
        if self.config["use_mixup"]:
            cutmix_or_mixup.append(
                v2.MixUp(
                    num_classes=self.num_classes,
                    alpha=self.config["alpha"],
                )
            )

        self.cutmix_or_mixup_aug = v2.RandomChoice(cutmix_or_mixup)

    def __call__(self, _x, _y) -> tuple[torch.Tensor]:
        r = torch.rand(1)
        prob = self.config["prob"]
        if r <= prob:
            if len(_y.shape) > 1:
                no_saliency_aug_idx = torch.where(_y >= 1)[0]
                if no_saliency_aug_idx.shape[0] > 1:
                    aug_only_by_cutmixup_x, aug_only_by_cutmixup_y = (
                        self.cutmix_or_mixup_aug(
                            _x[no_saliency_aug_idx], _y[no_saliency_aug_idx].argmax(
                                1)
                        )
                    )
                    _x[no_saliency_aug_idx] = aug_only_by_cutmixup_x
                    _y[no_saliency_aug_idx] = aug_only_by_cutmixup_y
            else:
                _x, _y = self.cutmix_or_mixup_aug(_x, _y)

        return _x, _y

    def get_x_y(self, aug_result):
        _x, _y = aug_result
        return _x, _y
