from .aug_base import *

# Implementation Credit: Pytorch


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
                v2.CutjMix(
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

    def __call__(self, _x, _y) -> AugResult:
        if torch.rand(1) <= self.config.get("prob", 1.0):
            if len(_y.shape) > 1:
                _y = _y.argmax(1)
            _x, _y = self.cutmix_or_mixup_aug(_x, _y)

        return AugResult(
            augmented_x=_x,
            augmented_y=_y
        )
