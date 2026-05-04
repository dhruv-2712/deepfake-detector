import io
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

_XCEPTION_MEAN = [0.5, 0.5, 0.5]
_XCEPTION_STD  = [0.5, 0.5, 0.5]


class RandomJPEGCompression:
    """
    Simulates re-encoding artifacts introduced when deepfakes are shared over
    social media. Applied after ToTensor so input is (C, H, W) in [0, 1].
    """
    def __init__(self, quality_low: int = 50, quality_high: int = 95):
        self.quality_low = quality_low
        self.quality_high = quality_high

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        quality = random.randint(self.quality_low, self.quality_high)
        pil = TF.to_pil_image(img.clamp(0, 1))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return TF.to_tensor(Image.open(buf).convert("RGB"))


class RandomGaussianNoise:
    """Additive Gaussian noise, mimicking sensor/quantisation noise."""
    def __init__(self, std_max: float = 0.05):
        self.std_max = std_max

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        std = random.uniform(0.0, self.std_max)
        return (img + torch.randn_like(img) * std).clamp(0, 1)


class TemporalFrameDrop:
    """
    Replaces randomly chosen frames with their successor — simulates the
    occasional dropped/duplicated frame from GAN video generation.
    Input: (T, C, H, W).
    """
    def __init__(self, drop_prob: float = 0.1):
        self.drop_prob = drop_prob

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[0]
        out = frames.clone()
        for t in range(T):
            if random.random() < self.drop_prob:
                out[t] = frames[(t + 1) % T]
        return out


class RandomHorizontalFlipVideo:
    """Flips all frames in a clip consistently. Input: (T, C, H, W)."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(frames, dims=[-1])
        return frames


def get_train_transforms(size: int = 299) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        RandomGaussianNoise(std_max=0.03),
        transforms.Normalize(mean=_XCEPTION_MEAN, std=_XCEPTION_STD),
    ])


def get_val_transforms(size: int = 299) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_XCEPTION_MEAN, std=_XCEPTION_STD),
    ])
