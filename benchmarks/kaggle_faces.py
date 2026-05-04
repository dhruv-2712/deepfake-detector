import random
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class KaggleFacesDataset(Dataset):
    """
    140k Real and Fake Faces dataset loader.

    Expected layout:
        root/real_vs_fake/real-vs-fake/
            train/real/*.jpg
            train/fake/*.jpg
            test/real/*.jpg
            test/fake/*.jpg

    Labels: 0 = real, 1 = fake.
    Returns: (image_tensor, label)
        image_tensor: (3, 224, 224) float32, ImageNet-normalized.

    The dataset has no validation split, so train is split 90/10 by default.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        transform=None,
    ):
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split!r}"

        base = Path(root) / "real_vs_fake" / "real-vs-fake"
        folder = base / ("test" if split == "test" else "train")

        samples: List[Tuple[Path, int]] = []
        for label, name in [(0, "real"), (1, "fake")]:
            samples.extend((p, label) for p in sorted((folder / name).glob("*.jpg")))

        if split in ("train", "val"):
            rng = random.Random(seed)
            rng.shuffle(samples)
            n_val = int(len(samples) * val_fraction)
            samples = samples[:n_val] if split == "val" else samples[n_val:]

        self.samples = samples
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label
