"""
train.py — trains the full deepfake detection pipeline.

Supports three modalities selectable via CLI:
  --modality image   : FaceForensics++ frames → ImageFeatureExtractor
  --modality audio   : ASVspoof waveforms     → AudioFeatureExtractor
  --modality video   : FaceForensics++ clips  → VideoFeatureExtractor

Usage:
    python train.py --modality image  --ffpp_root /data/FaceForensics++
    python train.py --modality audio  --asvspoof_root /data/ASVspoof2019
    python train.py --modality video  --ffpp_root /data/FaceForensics++
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor
from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE, VideoFeatureExtractor
from fusion.cross_attention import MultiModalFusionClassifier
from scripts.preextract import EmbeddingDataset
from utils.augmentations import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def weighted_bce(pred: torch.Tensor, target: torch.Tensor,
                 fake_weight: float, device: torch.device) -> torch.Tensor:
    pw = torch.tensor([fake_weight], device=device)
    w  = torch.where(target == 1, pw.expand_as(target), torch.ones_like(target))
    return (w * F.binary_cross_entropy(pred, target, reduction="none")).mean()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(img_ext, aud_ext, vid_ext, fusion, loader, device, modality):
    for m in (img_ext, aud_ext, vid_ext, fusion):
        m.eval()
    all_scores, all_labels = [], []

    for batch in loader:
        inputs, labels = batch

        if modality == "image":
            frames = inputs  # (B, T, 3, H, W) or (B, 3, H, W)
            if frames.ndim == 5:
                B, T, C, H, W = frames.shape
                emb = img_ext(frames.view(B * T, C, H, W).to(device))
                emb = emb.view(B, T, -1).mean(1)
            else:
                emb = img_ext(frames.to(device))
            scores = fusion(img_emb=emb).squeeze(1)

        elif modality == "audio":
            scores = fusion(aud_emb=aud_ext(inputs.to(device))).squeeze(1)

        elif modality == "video":
            clips = inputs.to(device)   # (B, T, C, H, W)
            scores = fusion(vid_emb=vid_ext(clips)).squeeze(1)

        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return {
        "auc": roc_auc_score(labels, scores),
        "ap":  average_precision_score(labels, scores),
        "eer": compute_eer(labels, scores),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_embeddings = bool(args.embeddings_dir)
    print(f"Device: {device}  |  Modality: {args.modality}  |  "
          f"Mode: {'cached embeddings' if use_embeddings else 'raw data'}")

    img_ext = ImageFeatureExtractor(pretrained=True).to(device)
    aud_ext = AudioFeatureExtractor(sample_rate=16000).to(device)
    vid_ext = VideoFeatureExtractor(pretrained=True).to(device)
    fusion  = MultiModalFusionClassifier().to(device)

    # Resume from checkpoint if provided
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        img_ext.load_state_dict(state.get("img_extractor", {}), strict=False)
        aud_ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
        vid_ext.load_state_dict(state.get("vid_extractor", {}), strict=False)
        fusion.load_state_dict(state.get("fusion", {}), strict=False)
        print(f"Resumed from {args.resume}")

    # When training on cached embeddings only the fusion head is updated.
    if use_embeddings:
        params = list(fusion.parameters())
    else:
        params = (
            list(img_ext.parameters())
            + list(aud_ext.parameters())
            + list(vid_ext.parameters())
            + list(fusion.parameters())
        )

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- datasets ---
    if use_embeddings:
        emb_dir   = Path(args.embeddings_dir)
        train_ds  = EmbeddingDataset(emb_dir / f"{args.modality}_train.npz")
        val_ds    = EmbeddingDataset(emb_dir / f"{args.modality}_val.npz")
    elif args.kaggle_root and args.modality == "image":
        from benchmarks.kaggle_faces import KaggleFacesDataset
        train_ds = KaggleFacesDataset(args.kaggle_root, split="train", transform=get_train_transforms())
        val_ds   = KaggleFacesDataset(args.kaggle_root, split="val",   transform=get_val_transforms())
    elif args.modality in ("image", "video"):
        if not args.ffpp_root:
            raise ValueError("--ffpp_root required for image/video modality")
        from benchmarks.ff_plusplus import FaceForensicsDataset
        train_ds = FaceForensicsDataset(
            args.ffpp_root, split="train",
            compression=args.compression,
            frames_per_video=CLIP_FRAMES if args.modality == "video" else args.frames_per_video,
            transform=get_train_transforms(),
        )
        val_ds = FaceForensicsDataset(
            args.ffpp_root, split="val",
            compression=args.compression,
            frames_per_video=CLIP_FRAMES if args.modality == "video" else args.frames_per_video,
            transform=get_val_transforms(),
        )
    else:
        if not args.asvspoof_root:
            raise ValueError("--asvspoof_root required for audio modality")
        from benchmarks.asvspoof import ASVspoofDataset
        train_ds = ASVspoofDataset(
            audio_dir=os.path.join(args.asvspoof_root, "flac"),
            protocol_file=os.path.join(args.asvspoof_root, "protocol", "train.txt"),
        )
        val_ds = ASVspoofDataset(
            audio_dir=os.path.join(args.asvspoof_root, "flac"),
            protocol_file=os.path.join(args.asvspoof_root, "protocol", "dev.txt"),
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        for m in (img_ext, aud_ext, vid_ext, fusion):
            m.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            target = labels.float().to(device)

            if use_embeddings:
                # inputs are already (B, 512) embeddings
                emb_kwargs = {f"{args.modality}_emb": inputs.to(device)}
                pred = fusion(**emb_kwargs).squeeze(1)

            elif args.modality == "image":
                frames = inputs
                if frames.ndim == 5:
                    B, T, C, H, W = frames.shape
                    emb = img_ext(frames.view(B * T, C, H, W).to(device))
                    emb = emb.view(B, T, -1).mean(1)
                else:
                    emb = img_ext(frames.to(device))
                pred = fusion(img_emb=emb).squeeze(1)

            elif args.modality == "audio":
                pred = fusion(aud_emb=aud_ext(inputs.to(device))).squeeze(1)

            elif args.modality == "video":
                pred = fusion(vid_emb=vid_ext(inputs.to(device))).squeeze(1)

            loss = weighted_bce(pred, target, args.fake_weight, device)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        metrics = evaluate(img_ext, aud_ext, vid_ext, fusion,
                           val_loader, device, args.modality)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={running_loss / len(train_loader):.4f}  "
            f"AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  EER={metrics['eer']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "modality": args.modality,
                    "img_extractor": img_ext.state_dict(),
                    "aud_extractor": aud_ext.state_dict(),
                    "vid_extractor": vid_ext.state_dict(),
                    "fusion": fusion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "auc": best_auc,
                },
                os.path.join(args.checkpoint_dir, "best.pt"),
            )
            print(f"  -> saved best checkpoint (AUC={best_auc:.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality",        choices=["image", "audio", "video"], default="image")
    parser.add_argument("--ffpp_root",       type=str, default=None)
    parser.add_argument("--asvspoof_root",   type=str, default=None)
    parser.add_argument("--compression",     type=str, default="c23")
    parser.add_argument("--frames_per_video",type=int, default=10)
    parser.add_argument("--epochs",          type=int, default=50)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--fake_weight",    type=float, default=2.0)
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--kaggle_root",    type=str,   default=None,
                        help="Root of 140k-real-and-fake-faces dataset")
    parser.add_argument("--embeddings_dir", type=str,   default=None,
                        help="Use pre-extracted embeddings (from scripts/preextract.py)")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)
