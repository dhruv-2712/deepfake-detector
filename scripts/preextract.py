"""
Pre-extract feature embeddings to disk so training only fine-tunes the
fusion head — 10-50x faster than re-running the backbone each epoch.

Usage:
    python scripts/preextract.py \\
        --checkpoint checkpoints/best.pt \\
        --ffpp_root  /data/FaceForensics++ \\
        --modality   image \\
        --output_dir embeddings/

Output files (compressed numpy):
    embeddings/image_train.npz   →  {'embeddings': (N, 512), 'labels': (N,)}
    embeddings/image_val.npz
    embeddings/image_test.npz

Then pass --embeddings_dir embeddings/ to train.py to use cached features.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor
from detectors.video.extractor import CLIP_FRAMES, VideoFeatureExtractor
from utils.augmentations import get_val_transforms


# ---------------------------------------------------------------------------
# Dataset for pre-extracted embeddings
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Loads pre-extracted embeddings from a .npz file produced by this script.
    Drop-in replacement for FaceForensicsDataset / ASVspoofDataset when
    --embeddings_dir is passed to train.py.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.embeddings = torch.from_numpy(data["embeddings"].astype(np.float32))
        self.labels     = data["labels"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], int(self.labels[idx])


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_ffpp(extractor, loader, device, modality):
    extractor.eval()
    embs, labs = [], []

    for inputs, labels in loader:
        inputs = inputs.to(device)

        if modality == "image":
            if inputs.ndim == 5:               # (B, T, C, H, W)
                B, T, C, H, W = inputs.shape
                e = extractor(inputs.view(B * T, C, H, W)).view(B, T, -1).mean(1)
            else:
                e = extractor(inputs)
        elif modality == "video":
            e = extractor(inputs)             # (B, T, C, H, W)
        elif modality == "audio":
            e = extractor(inputs)

        embs.append(e.cpu().numpy())
        labs.append(labels.numpy())
        print(".", end="", flush=True)

    print()
    return np.concatenate(embs), np.concatenate(labs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--modality",         choices=["image", "audio", "video"], default="image")
    parser.add_argument("--ffpp_root",        default=None)
    parser.add_argument("--asvspoof_root",    default=None)
    parser.add_argument("--compression",      default="c23")
    parser.add_argument("--frames_per_video", type=int, default=10)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--workers",          type=int, default=4)
    parser.add_argument("--output_dir",       default="embeddings")
    parser.add_argument("--face_crop",        action="store_true",
                        help="Use MTCNN face detection when loading FF++ frames.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load extractor
    state = torch.load(args.checkpoint, map_location=device)
    if args.modality == "image":
        ext = ImageFeatureExtractor(pretrained=False).to(device)
        ext.load_state_dict(state.get("img_extractor", {}), strict=False)
    elif args.modality == "audio":
        ext = AudioFeatureExtractor(sample_rate=16000).to(device)
        ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
    elif args.modality == "video":
        ext = VideoFeatureExtractor(pretrained=False).to(device)
        ext.load_state_dict(state.get("vid_extractor", {}), strict=False)

    for split in ("train", "val", "test"):
        out_path = Path(args.output_dir) / f"{args.modality}_{split}.npz"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        if args.modality in ("image", "video"):
            if not args.ffpp_root:
                raise ValueError("--ffpp_root required")
            from benchmarks.ff_plusplus import FaceForensicsDataset
            fpv = CLIP_FRAMES if args.modality == "video" else args.frames_per_video
            ds_kwargs = dict(
                split=split,
                compression=args.compression,
                frames_per_video=fpv,
            )
            if args.modality == "image":
                ds_kwargs["transform"] = get_val_transforms(size=299)
            elif args.modality == "video":
                from torchvision import transforms as _T
                ds_kwargs["transform"] = _T.Compose([
                    _T.Resize((CLIP_SIZE, CLIP_SIZE)),
                    _T.ToTensor(),
                    _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            ds_kwargs["use_face_crop"] = args.face_crop
            ds = FaceForensicsDataset(args.ffpp_root, **ds_kwargs)
        else:
            if not args.asvspoof_root:
                raise ValueError("--asvspoof_root required")
            from benchmarks.asvspoof import ASVspoofDataset
            import os
            proto_map = {"train": "train.txt", "val": "dev.txt", "test": "eval.txt"}
            ds = ASVspoofDataset(
                audio_dir=os.path.join(args.asvspoof_root, "flac"),
                protocol_file=os.path.join(args.asvspoof_root, "protocol", proto_map[split]),
            )

        if len(ds) == 0:
            print(f"[skip] {split} — no samples found")
            continue

        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        print(f"Extracting {split} ({len(ds)} samples) ", end="")
        embs, labs = extract_ffpp(ext, loader, device, args.modality)
        np.savez_compressed(out_path, embeddings=embs, labels=labs)
        print(f"  saved → {out_path}  shape={embs.shape}")


if __name__ == "__main__":
    main()
