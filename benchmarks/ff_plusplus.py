import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

_SPLIT_SIZES = {"train": 720, "val": 140, "test": 140}

_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ dataset loader.

    Expected layout:
        root/
          original_sequences/youtube/{compression}/videos/*.mp4
          manipulated_sequences/{manipulation}/{compression}/videos/*.mp4

    Labels: 0 = real, 1 = fake.
    Returns: (frames_tensor, label)
        frames_tensor: (T, 3, 224, 224) float32, ImageNet-normalized.

    Args:
        manipulations: subset of MANIPULATIONS to load. None = all four.
    """

    def __init__(
        self,
        root: str,
        compression: str = "c23",
        split: str = "train",
        frames_per_video: int = 10,
        seed: int = 42,
        manipulations: Optional[List[str]] = None,
        transform=None,
    ):
        assert split in _SPLIT_SIZES, f"split must be one of {list(_SPLIT_SIZES)}"
        self.root = Path(root)
        self.compression = compression
        self.split = split
        self.frames_per_video = frames_per_video
        self.seed = seed
        self.manipulations = manipulations or MANIPULATIONS
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            _NORMALIZE,
        ])

        self.samples: List[Tuple[Path, int]] = []
        self._build_index()

    def _split_videos(self, videos: List[Path]) -> List[Path]:
        videos = sorted(videos, key=lambda p: p.stem)
        n_tr, n_val = _SPLIT_SIZES["train"], _SPLIT_SIZES["val"]
        slices = {
            "train": videos[:n_tr],
            "val":   videos[n_tr: n_tr + n_val],
            "test":  videos[n_tr + n_val:],
        }
        return slices[self.split]

    def _build_index(self):
        real_dir = (
            self.root / "original_sequences" / "youtube" / self.compression / "videos"
        )
        if real_dir.exists():
            for v in self._split_videos(sorted(real_dir.glob("*.mp4"))):
                self.samples.append((v, 0))

        for manip in self.manipulations:
            fake_dir = (
                self.root / "manipulated_sequences" / manip / self.compression / "videos"
            )
            if fake_dir.exists():
                for v in self._split_videos(sorted(fake_dir.glob("*.mp4"))):
                    self.samples.append((v, 1))

    def _extract_frames(self, video_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        rng = random.Random(self.seed + hash(video_path.stem))
        indices = sorted(rng.sample(range(total), min(self.frames_per_video, total)))

        raw_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        placeholder = np.zeros((4, 4, 3), dtype=np.uint8)
        while len(raw_frames) < self.frames_per_video:
            raw_frames.append(raw_frames[-1] if raw_frames else placeholder)

        return torch.stack([self.transform(Image.fromarray(f)) for f in raw_frames])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        return frames, label
