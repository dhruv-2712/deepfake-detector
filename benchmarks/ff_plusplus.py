import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]

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
        frames_tensor: (T, 3, H, H) float32, normalized per transform.

    Args:
        manipulations:  subset of MANIPULATIONS to load. None = all five.
        random_frames:  sample different frames each call (for training augmentation).
                        When False, frames are deterministic per video (for val/test).
        use_face_crop:  run MTCNN face detection and crop to the largest face before
                        applying the transform. Requires facenet-pytorch. Slower but
                        focuses the model on the face region.
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
        random_frames: bool = False,
        use_face_crop: bool = False,
    ):
        assert split in _SPLIT_SIZES, f"split must be one of {list(_SPLIT_SIZES)}"
        self.root = Path(root)
        self.compression = compression
        self.split = split
        self.frames_per_video = frames_per_video
        self.seed = seed
        self.manipulations = manipulations or MANIPULATIONS
        self.random_frames = random_frames
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._detector = None
        if use_face_crop:
            try:
                from utils.face_align import FaceDetector
                self._detector = FaceDetector()
            except ImportError:
                print("Warning: facenet-pytorch not installed — use_face_crop disabled.")

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

        if self.random_frames:
            indices = sorted(random.sample(range(total), min(self.frames_per_video, total)))
        else:
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

        tensors = []
        for f in raw_frames:
            pil = Image.fromarray(f)
            if self._detector is not None:
                cropped = self._detector.detect_largest(pil, size=299)
                if cropped is not None:
                    pil = cropped
            tensors.append(self.transform(pil))
        return torch.stack(tensors)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        return frames, label
