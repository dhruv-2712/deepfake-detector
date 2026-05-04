"""
eval.py — evaluates a trained checkpoint on test splits.

FF++ (per-manipulation breakdown):
    python eval.py --checkpoint checkpoints/best.pt \\
                   --ffpp_root /data/FaceForensics++ --modality image

ASVspoof:
    python eval.py --checkpoint checkpoints/best_audio.pt \\
                   --asvspoof_root /data/ASVspoof2019 --modality audio

Fusion (pre-extracted embeddings):
    python eval.py --checkpoint checkpoints/best_fusion.pt \\
                   --embeddings_dir embeddings/ --modality fusion

Results are saved to results/eval_<timestamp>.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from benchmarks.ff_plusplus import MANIPULATIONS, FaceForensicsDataset
from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor
from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE, VideoFeatureExtractor
from fusion.cross_attention import MultiModalFusionClassifier
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
            if inputs.ndim == 5:
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


@torch.no_grad()
def run_fusion_inference(embeddings_dir: str, split: str, fusion, device,
                         batch_size: int) -> tuple:
    """Load pre-extracted embeddings and score them through the fusion model."""
    d = Path(embeddings_dir)
    img_emb = aud_emb = vid_emb = None
    labels = None

    for mod in ("image", "audio", "video"):
        p = d / f"{mod}_{split}.npz"
        if not p.exists():
            continue
        data = np.load(p)
        t = torch.from_numpy(data["embeddings"].astype(np.float32))
        if mod == "image":   img_emb = t
        elif mod == "audio": aud_emb = t
        elif mod == "video": vid_emb = t
        if labels is None:
            labels = data["labels"].astype(np.int64)
        print(f"  Loaded {p.name}  n={len(t)}")

    if labels is None:
        raise FileNotFoundError(f"No embeddings found in {d} for split={split}")

    n = len(labels)
    all_scores = []
    fusion.eval()
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        ie = img_emb[sl].to(device) if img_emb is not None else None
        ae = aud_emb[sl].to(device) if aud_emb is not None else None
        ve = vid_emb[sl].to(device) if vid_emb is not None else None
        logits = fusion(img_emb=ie, aud_emb=ae, vid_emb=ve)
        all_scores.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())

    return np.concatenate(all_scores), labels


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------

def eval_ffpp(args, img_ext, aud_ext, vid_ext, head, device) -> dict:
    results = {}
    all_scores, all_labels = [], []

    fpv = CLIP_FRAMES if args.modality == "video" else args.frames_per_video

    vid_tfm = None
    if args.modality == "video":
        from torchvision import transforms as T
        vid_tfm = T.Compose([
            T.Resize((CLIP_SIZE, CLIP_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    for manip in MANIPULATIONS:
        ds = FaceForensicsDataset(
            args.ffpp_root,
            compression=args.compression,
            split=args.split,
            frames_per_video=fpv,
            manipulations=[manip],
            transform=vid_tfm if vid_tfm else get_val_transforms(size=299),
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
    proto_map = {"val": "dev.txt", "test": "eval.txt"}
    protocol  = os.path.join(args.asvspoof_root, "protocol", proto_map[args.split])
    ds = ASVspoofDataset(
        audio_dir=os.path.join(args.asvspoof_root, "flac"),
        protocol_file=protocol,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    scores, labels = run_inference(loader, img_ext, aud_ext, vid_ext, head, device, "audio")
    return {"overall": compute_metrics(labels, scores)}


def eval_fusion(args, fusion, device) -> dict:
    scores, labels = run_fusion_inference(
        args.embeddings_dir, args.split, fusion, device, args.batch_size
    )
    return {"overall": compute_metrics(labels, scores)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: dict, title: str):
    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Category':<20}  {'AUC':>6}  {'AP':>6}  {'EER':>6}  {'N':>6}")
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
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--ffpp_root",        default=None)
    parser.add_argument("--asvspoof_root",    default=None)
    parser.add_argument("--embeddings_dir",   default=None,
                        help="Pre-extracted embeddings dir (required for --modality fusion).")
    parser.add_argument("--modality",         choices=["image", "audio", "video", "fusion"],
                        default="image")
    parser.add_argument("--split",            default="test", choices=["val", "test"])
    parser.add_argument("--compression",      default="c23")
    parser.add_argument("--frames_per_video", type=int, default=10)
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--workers",          type=int, default=4)
    parser.add_argument("--output_dir",       default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    state = torch.load(args.checkpoint, map_location=device)
    val_info = f"  val_acc={state['val_acc']:.4f}" if "val_acc" in state else ""
    print(f"Loaded: {args.checkpoint}  (epoch {state.get('epoch', '?')}){val_info}")

    # ------------------------------------------------------------------
    # Fusion evaluation (embeddings path)
    # ------------------------------------------------------------------
    if args.modality == "fusion":
        if not args.embeddings_dir:
            parser.error("--embeddings_dir required for --modality fusion")
        fusion_state = state.get("fusion", {})
        if not fusion_state:
            parser.error("Checkpoint has no fusion weights. "
                         "Train with --embeddings_dir first.")
        fusion = MultiModalFusionClassifier().to(device)
        fusion.load_state_dict(fusion_state)
        results = eval_fusion(args, fusion, device)
        title = (f"Fusion | split={args.split} | "
                 f"embeddings={args.embeddings_dir}")
        print_table(results, title)

    # ------------------------------------------------------------------
    # Single-modality evaluation
    # ------------------------------------------------------------------
    else:
        img_ext = ImageFeatureExtractor(pretrained=False).to(device)
        aud_ext = AudioFeatureExtractor(sample_rate=16000).to(device)
        vid_ext = VideoFeatureExtractor(pretrained=False).to(device)
        head    = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        ).to(device)

        img_ext.load_state_dict(state.get("img_extractor", {}), strict=False)
        aud_ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
        vid_ext.load_state_dict(state.get("vid_extractor", {}), strict=False)
        if state.get("head"):
            head.load_state_dict(state["head"], strict=False)

        if args.ffpp_root:
            results = eval_ffpp(args, img_ext, aud_ext, vid_ext, head, device)
            title = (f"FF++ | split={args.split} | compression={args.compression} "
                     f"| modality={args.modality}")
            print_table(results, title)
        elif args.asvspoof_root:
            results = eval_asvspoof(args, img_ext, aud_ext, vid_ext, head, device)
            title = f"ASVspoof | split={args.split} | modality=audio"
            print_table(results, title)
        else:
            parser.error("Provide --ffpp_root, --asvspoof_root, or --embeddings_dir")

    # Save JSON
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
