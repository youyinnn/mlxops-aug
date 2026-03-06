# mlxops-aug

Data augmentation methods for deep learning, including CutMix, MixUp, and PuzzleMix.

## Installation

```bash
pip install mlxops-aug
```

For PuzzleMix support (requires `pygco`):

```bash
pip install "mlxops-aug[puzzlemix]"
```

> **Note:** `pygco` requires a C compiler. See [pyGCO](https://github.com/Borda/pyGCO) for installation instructions.

## Usage

### CutMix / MixUp

```python
from mlxops_aug import CutMixUp

aug = CutMixUp(
    num_classes=10,
    config={
        "use_cutmix": True,
        "use_mixup": True,
        "alpha": 1.0,
        "prob": 0.5,
    },
)

x_aug, y_aug = aug(x, y)
```

### PuzzleMix

```python
from mlxops_aug import PuzzleMix

aug = PuzzleMix(
    num_classes=10,
    config={
        "block_num": 2,
        "transport": True,
        "prob": 1.0,
    },
)
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15

## License

MIT
