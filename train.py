"""
train.py

Image-only (end-to-end backbone fine-tuning):
    python train.py --ffpp_root /data/FaceForensics++

Multimodal (fast path on pre-extracted embeddings):
    python train.py --embeddings_dir embeddings/
    # Run scripts/preextract.py first for each modality.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from benchmarks.ff_plusplus import MANIPULATIONS, FaceForensicsDataset
from detectors.image.extractor import ImageFeatureExtractor
from fusion.cross_attention import MultiModalFusionClassifier
from utils.augmentations import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Multimodal embedding dataset (fast path: pre-extracted features)
# ---------------------------------------------------------------------------

class MultiModalEmbeddingDataset(Dataset):
    """
    Loads pre-extracted per-modality .npz files produced by scripts/preextract.py.
    Missing modalities are returned as zero tensors so the DataLoader can collate normally.
    Returns (img_emb, aud_emb, vid_emb, label).
    """

    def __init__(self, embeddings_dir: str, split: str):
        d = Path(embeddings_dir)
        self.img_emb = self.aud_emb = self.vid_emb = None
        labels = None

        for mod, attr in [("image", "img_emb"), ("audio", "aud_emb"), ("video", "vid_emb")]:
            p = d / f"{mod}_{split}.npz"
            if p.exists():
                data = np.load(p)
                emb = torch.from_numpy(data["embeddings"].astype(np.float32))
                setattr(self, attr, emb)
                if labels is None:
                    labels = data["labels"].astype(np.int64)
                print(f"  Loaded {p.name}  shape={emb.shape}")

        if labels is None:
            raise FileNotFoundError(
                f"No embedding .npz files found in {d} for split={split}. "
                "Run scripts/preextract.py first."
            )
        self.labels = labels
        self._zero = torch.zeros(512)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.img_emb[idx] if self.img_emb is not None else self._zero
        aud = self.aud_emb[idx] if self.aud_emb is not None else self._zero
        vid = self.vid_emb[idx] if self.vid_emb is not None else self._zero
        return img, aud, vid, int(self.labels[idx])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_image_epoch(loader, img_ext, head, criterion, device, optimizer=None):
    training = optimizer is not None
    img_ext.train(training)
    head.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for frames, labels in loader:
            labels = labels.float().to(device)
            B, T, C, H, W = frames.shape
            emb = img_ext(frames.view(B * T, C, H, W).to(device)).view(B, T, -1).mean(1)
            preds = head(emb).squeeze(1)
            loss = criterion(preds, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0).float() == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


def run_fusion_epoch(loader, fusion, criterion, device, optimizer=None):
    training = optimizer is not None
    fusion.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img_emb, aud_emb, vid_emb, labels in loader:
            labels  = labels.float().to(device)
            img_emb = img_emb.to(device)
            aud_emb = aud_emb.to(device)
            vid_emb = vid_emb.to(device)
            preds = fusion(img_emb=img_emb, aud_emb=aud_emb, vid_emb=vid_emb).squeeze(1)
            loss = criterion(preds, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0).float() == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


def save_checkpoint(path, epoch, val_acc, val_loss, img_ext=None, fusion=None, head=None):
    torch.save(
        {
            "epoch":         epoch,
            "val_acc":       round(val_acc, 6),
            "val_loss":      round(val_loss, 6),
            "img_extractor": img_ext.state_dict() if img_ext is not None else {},
            "aud_extractor": {},
            "vid_extractor": {},
            "fusion":        fusion.state_dict() if fusion is not None else {},
            "head":          head.state_dict() if head is not None else {},
        },
        path,
    )


def _make_sampler(label_list):
    n_real = label_list.count(0)
    n_fake = label_list.count(1)
    weights = [1.0 / n_real if l == 0 else 1.0 / n_fake for l in label_list]
    return WeightedRandomSampler(weights, len(weights))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    # data sources (mutually exclusive)
    parser.add_argument("--ffpp_root",         default=None,
                        help="FF++ root dir — image end-to-end training.")
    parser.add_argument("--embeddings_dir",    default=None,
                        help="Pre-extracted embeddings dir — fast multimodal training.")
    # FF++ options (image path only)
    parser.add_argument("--compression",       default="c23")
    parser.add_argument("--frames_per_video",  type=int, default=4)
    parser.add_argument("--manipulations",     nargs="+", default=None,
                        help="FF++ manipulations. Defaults to all four.")
    parser.add_argument("--val_split",         type=float, default=0.2)
    # training hyperparams
    parser.add_argument("--epochs",            type=int, default=20)
    parser.add_argument("--batch_size",        type=int, default=8)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--workers",           type=int, default=0)
    parser.add_argument("--checkpoint_dir",    default="checkpoints")
    parser.add_argument("--patience",          type=int, default=5)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not args.ffpp_root and not args.embeddings_dir:
        raise ValueError("Provide --ffpp_root (image training) or --embeddings_dir (multimodal).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fast multimodal path: train fusion head on pre-extracted embeddings
    # ------------------------------------------------------------------
    if args.embeddings_dir:
        print(f"\nMultimodal embedding path: {args.embeddings_dir}")
        print("Loading train embeddings...")
        train_ds = MultiModalEmbeddingDataset(args.embeddings_dir, "train")
        print("Loading val embeddings...")
        val_ds   = MultiModalEmbeddingDataset(args.embeddings_dir, "val")
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=_make_sampler(train_ds.labels.tolist()),
            num_workers=args.workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        fusion    = MultiModalFusionClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(fusion.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"LR={args.lr:.2e}  Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")

        best_val_acc = 0.0
        epochs_no_improve = 0
        ckpt_path = os.path.join(args.checkpoint_dir, "best_fusion.pt")

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_fusion_epoch(train_loader, fusion, criterion, device, optimizer)
            vl_loss, vl_acc = run_fusion_epoch(val_loader,   fusion, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                epochs_no_improve = 0
                save_checkpoint(ckpt_path, epoch, vl_acc, vl_loss, fusion=fusion)
                print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Done. Best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")
        return

    # ------------------------------------------------------------------
    # Image path: fine-tune image extractor end-to-end on FF++
    # ------------------------------------------------------------------
    manips = args.manipulations or MANIPULATIONS
    print(f"\nImage path | manipulations: {manips}")

    ds_kwargs = dict(
        compression=args.compression,
        split="train",
        frames_per_video=args.frames_per_video,
        manipulations=manips,
    )

    # Build train/val split from the FF++ "train" split
    full_ds = FaceForensicsDataset(args.ffpp_root, **ds_kwargs)
    all_samples = list(full_ds.samples)
    random.Random(42).shuffle(all_samples)
    n_val        = max(1, int(len(all_samples) * args.val_split))
    val_samples  = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    train_ds = FaceForensicsDataset(args.ffpp_root, **ds_kwargs,
                                    transform=get_train_transforms(size=299))
    train_ds.samples = train_samples
    val_ds = FaceForensicsDataset(args.ffpp_root, **ds_kwargs,
                                  transform=get_val_transforms(size=299))
    val_ds.samples = val_samples
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=_make_sampler([lbl for _, lbl in train_samples]),
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    img_ext = ImageFeatureExtractor(pretrained=True).to(device)
    head = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW([
        {"params": img_ext.backbone.parameters(),
         "lr": args.lr * 0.1},
        {"params": [p for n, p in img_ext.named_parameters() if "backbone" not in n],
         "lr": args.lr},
        {"params": head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"LR head/srm={args.lr:.2e}  backbone={args.lr * 0.1:.2e}")

    best_val_acc = 0.0
    epochs_no_improve = 0
    ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_image_epoch(train_loader, img_ext, head, criterion, device, optimizer)
        vl_loss, vl_acc = run_image_epoch(val_loader,   img_ext, head, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            save_checkpoint(ckpt_path, epoch, vl_acc, vl_loss, img_ext=img_ext, head=head)
            print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")


if __name__ == "__main__":
    main()
