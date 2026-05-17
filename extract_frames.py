"""
One-time frame extraction from FF++ videos to JPEG.

Produces a folder layout compatible with KaggleFacesDataset so you can pass
--kaggle_root to train.py without any other changes.

Usage:
    python extract_frames.py --ffpp_root data/ffpp --out data/ffpp_frames --compression c40
    python extract_frames.py --ffpp_root data/ffpp --out data/ffpp_frames --compression c40 --frames 8

After running, train with:
    python train.py --kaggle_root data/ffpp_frames --checkpoint_dir checkpoints/ffpp --fake_weight 5.0
"""
import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
_SPLIT_SIZES = {"train": 720, "val": 140, "test": 140}


def split_videos(videos, split):
    videos = sorted(videos, key=lambda p: p.stem)
    n_tr, n_val = _SPLIT_SIZES["train"], _SPLIT_SIZES["val"]
    slices = {
        "train": videos[:n_tr],
        "val":   videos[n_tr: n_tr + n_val],
        "test":  videos[n_tr + n_val:],
    }
    return slices[split]


def extract(video_path: Path, out_dir: Path, n_frames: int, seed: int):
    cap = cv2.VideoCapture(str(video_path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    rng = random.Random(seed + hash(video_path.stem))
    indices = sorted(rng.sample(range(total), min(n_frames, total)))

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = out_dir / f"{video_path.stem}_f{idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffpp_root",   required=True)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--compression", default="c40")
    parser.add_argument("--frames",      type=int, default=8,
                        help="Frames to extract per video (default 8)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    ffpp  = Path(args.ffpp_root)
    # Output layout matches KaggleFacesDataset exactly
    base  = Path(args.out) / "real_vs_fake" / "real-vs-fake"

    for split in ("train", "test"):
        (base / split / "real").mkdir(parents=True, exist_ok=True)
        (base / split / "fake").mkdir(parents=True, exist_ok=True)

    total_saved = 0

    # Real videos
    real_dir = ffpp / "original_sequences" / "youtube" / args.compression / "videos"
    if not real_dir.exists():
        print(f"[WARN] Real dir not found: {real_dir}")
    else:
        for split in ("train", "test"):
            videos = split_videos(list(real_dir.glob("*.mp4")), split)
            out_dir = base / split / "real"
            for v in tqdm(videos, desc=f"real/{split}"):
                total_saved += extract(v, out_dir, args.frames, args.seed)

    # Fake videos
    for manip in MANIPULATIONS:
        fake_dir = ffpp / "manipulated_sequences" / manip / args.compression / "videos"
        if not fake_dir.exists():
            print(f"[WARN] Fake dir not found: {fake_dir}")
            continue
        for split in ("train", "test"):
            videos = split_videos(list(fake_dir.glob("*.mp4")), split)
            out_dir = base / split / "fake"
            for v in tqdm(videos, desc=f"{manip}/{split}"):
                total_saved += extract(v, out_dir, args.frames, args.seed)

    print(f"\nDone. {total_saved} JPEG frames saved to {base}")
    real_tr = len(list((base / 'train' / 'real').glob('*.jpg')))
    fake_tr = len(list((base / 'train' / 'fake').glob('*.jpg')))
    real_te = len(list((base / 'test'  / 'real').glob('*.jpg')))
    fake_te = len(list((base / 'test'  / 'fake').glob('*.jpg')))
    print(f"  train: {real_tr} real  {fake_tr} fake")
    print(f"  test:  {real_te} real  {fake_te} fake")
    print(f"\nNow train with:")
    print(f"  python train.py --kaggle_root {args.out} --checkpoint_dir checkpoints/ffpp --fake_weight 5.0 --workers 4")


if __name__ == "__main__":
    main()
