"""
eval.py — evaluates a trained checkpoint on test splits.

FF++ (per-manipulation breakdown):
    python eval.py --checkpoint checkpoints/best.pt \\
                   --ffpp_root /data/FaceForensics++ --modality image

ASVspoof:
    python eval.py --checkpoint checkpoints/best.pt \\
                   --asvspoof_root /data/ASVspoof2019 --modality audio

Results are saved to results/eval_<timestamp>.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

import torch.nn as nn

from benchmarks.ff_plusplus import MANIPULATIONS, FaceForensicsDataset
from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor
from detectors.video.extractor import CLIP_FRAMES, VideoFeatureExtractor
from utils.augmentations import get_val_transforms


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "eer": float("nan"), "n": len(labels)}
    return {
        "auc": round(roc_auc_score(labels, scores), 4),
        "ap":  round(average_precision_score(labels, scores), 4),
        "eer": round(compute_eer(labels, scores), 4),
        "n":   int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(loader, img_ext, aud_ext, vid_ext, head, device, modality):
    for m in (img_ext, aud_ext, vid_ext, head):
        m.eval()
    all_scores, all_labels = [], []

    for inputs, labels in loader:
        inputs = inputs.to(device)

        if modality == "image":
            if inputs.ndim == 5:                         # (B, T, C, H, W)
                B, T, C, H, W = inputs.shape
                emb = img_ext(inputs.view(B * T, C, H, W)).view(B, T, -1).mean(1)
            else:
                emb = img_ext(inputs)
        elif modality == "audio":
            emb = aud_ext(inputs)
        elif modality == "video":
            emb = vid_ext(inputs)

        logits = head(emb).squeeze(1)
        scores = torch.sigmoid(logits)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------

def eval_ffpp(args, img_ext, aud_ext, vid_ext, head, device) -> dict:
    results = {}
    all_scores, all_labels = [], []

    fpv = CLIP_FRAMES if args.modality == "video" else args.frames_per_video
    transform = get_val_transforms(size=299)

    for manip in MANIPULATIONS:
        ds = FaceForensicsDataset(
            args.ffpp_root,
            compression=args.compression,
            split=args.split,
            frames_per_video=fpv,
            manipulations=[manip],
            transform=transform,
        )
        if len(ds) == 0:
            print(f"  [skip] {manip} — no data found")
            continue

        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        scores, labels = run_inference(loader, img_ext, aud_ext, vid_ext, head, device, args.modality)
        results[manip] = compute_metrics(labels, scores)
        all_scores.append(scores)
        all_labels.append(labels)

    if all_scores:
        results["overall"] = compute_metrics(
            np.concatenate(all_labels), np.concatenate(all_scores)
        )

    return results


def eval_asvspoof(args, img_ext, aud_ext, vid_ext, head, device) -> dict:
    from benchmarks.asvspoof import ASVspoofDataset

    protocol = os.path.join(args.asvspoof_root, "protocol",
                            "eval.txt" if args.split == "test" else f"{args.split}.txt")
    ds = ASVspoofDataset(
        audio_dir=os.path.join(args.asvspoof_root, "flac"),
        protocol_file=protocol,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    scores, labels = run_inference(loader, img_ext, aud_ext, vid_ext, head, device, "audio")
    return {"overall": compute_metrics(labels, scores)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: dict, title: str):
    print(f"\n{title}")
    print("=" * 60)
    header = f"{'Category':<20}  {'AUC':>6}  {'AP':>6}  {'EER':>6}  {'N':>6}"
    print(header)
    print("-" * 60)
    for name, m in results.items():
        if name == "overall":
            continue
        print(f"  {name:<18}  {m['auc']:>6.4f}  {m['ap']:>6.4f}  {m['eer']:>6.4f}  {m['n']:>6}")
    if "overall" in results:
        print("-" * 60)
        m = results["overall"]
        print(f"  {'Overall':<18}  {m['auc']:>6.4f}  {m['ap']:>6.4f}  {m['eer']:>6.4f}  {m['n']:>6}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--ffpp_root",       default=None)
    parser.add_argument("--asvspoof_root",   default=None)
    parser.add_argument("--modality",        choices=["image", "audio", "video"], default="image")
    parser.add_argument("--split",           default="test", choices=["val", "test"])
    parser.add_argument("--compression",     default="c23")
    parser.add_argument("--frames_per_video",type=int, default=10)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--workers",         type=int, default=4)
    parser.add_argument("--output_dir",      default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    img_ext = ImageFeatureExtractor(pretrained=False).to(device)
    aud_ext = AudioFeatureExtractor(sample_rate=16000).to(device)
    vid_ext = VideoFeatureExtractor(pretrained=False).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    img_ext.load_state_dict(state.get("img_extractor", {}), strict=False)
    aud_ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
    vid_ext.load_state_dict(state.get("vid_extractor", {}), strict=False)

    head_state = state.get("head", {})
    head = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    ).to(device)
    if head_state:
        head.load_state_dict(head_state, strict=False)
    val_info = f"  val_acc={state['val_acc']:.4f}" if "val_acc" in state else ""
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {state.get('epoch', '?')}){val_info}")

    # Run evaluation
    if args.ffpp_root:
        results = eval_ffpp(args, img_ext, aud_ext, vid_ext, head, device)
        title = f"FF++ | split={args.split} | compression={args.compression} | modality={args.modality}"
        print_table(results, title)
    elif args.asvspoof_root:
        results = eval_asvspoof(args, img_ext, aud_ext, vid_ext, head, device)
        title = f"ASVspoof | split={args.split} | modality=audio"
        print_table(results, title)
    else:
        parser.error("Provide --ffpp_root or --asvspoof_root")

    # Save JSON
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
