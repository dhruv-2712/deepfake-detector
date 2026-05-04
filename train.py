"""
train.py

Image-only (end-to-end backbone fine-tuning on FF++):
    python train.py --ffpp_root /data/FaceForensics++

Audio-only (train AudioFeatureExtractor on ASVspoof):
    python train.py --asvspoof_root /data/ASVspoof2019

Multimodal (fast fusion path on pre-extracted embeddings):
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
from detectors.audio.extractor import AudioFeatureExtractor
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
    has_img / has_aud / has_vid indicate which modalities have real data.
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
        self.has_img = self.img_emb is not None
        self.has_aud = self.aud_emb is not None
        self.has_vid = self.vid_emb is not None
        self._zero = torch.zeros(512)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.img_emb[idx] if self.has_img else self._zero
        aud = self.aud_emb[idx] if self.has_aud else self._zero
        vid = self.vid_emb[idx] if self.has_vid else self._zero
        return img, aud, vid, int(self.labels[idx])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _dropout_modalities(img, aud, vid, p):
    """Randomly None-out present modalities to train missing_token, keeping at least one."""
    mods = [img, aud, vid]
    present = [i for i, m in enumerate(mods) if m is not None]
    if len(present) <= 1 or p <= 0:
        return img, aud, vid
    for i in present:
        if random.random() < p and sum(m is not None for m in mods) > 1:
            mods[i] = None
    return tuple(mods)


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


def run_generic_epoch(loader, extractor, head, criterion, device, optimizer=None):
    """Generic epoch runner for single-input extractors (audio, etc.)."""
    training = optimizer is not None
    extractor.train(training)
    head.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            labels = labels.float().to(device)
            emb = extractor(inputs.to(device))
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


def run_fusion_epoch(loader, fusion, criterion, device, optimizer=None,
                     has_img=True, has_aud=True, has_vid=True, mod_dropout=0.15):
    """
    Fusion epoch. During training, randomly None-out present modalities (mod_dropout)
    so the fusion model learns to use missing_token for absent inputs.
    """
    training = optimizer is not None
    fusion.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img_t, aud_t, vid_t, labels in loader:
            labels  = labels.float().to(device)
            img_emb = img_t.to(device) if has_img else None
            aud_emb = aud_t.to(device) if has_aud else None
            vid_emb = vid_t.to(device) if has_vid else None
            if training:
                img_emb, aud_emb, vid_emb = _dropout_modalities(
                    img_emb, aud_emb, vid_emb, mod_dropout
                )
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


def save_checkpoint(path, epoch, val_acc, val_loss,
                    img_ext=None, aud_ext=None, fusion=None, head=None):
    torch.save(
        {
            "epoch":         epoch,
            "val_acc":       round(val_acc, 6),
            "val_loss":      round(val_loss, 6),
            "img_extractor": img_ext.state_dict() if img_ext is not None else {},
            "aud_extractor": aud_ext.state_dict() if aud_ext is not None else {},
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


def _make_head(device):
    return nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    ).to(device)


def _training_loop(train_loader, val_loader, model_fn, criterion, device,
                   optimizer, scheduler, args, ckpt_path, save_fn):
    """Shared early-stopping loop. save_fn(path, epoch, val_acc, val_loss) saves the checkpoint."""
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = model_fn(train_loader, criterion, device, optimizer)
        vl_loss, vl_acc = model_fn(val_loader,   criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            save_fn(ckpt_path, epoch, vl_acc, vl_loss)
            print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_val_acc


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    # data sources
    parser.add_argument("--ffpp_root",       default=None,
                        help="FF++ root — image end-to-end training.")
    parser.add_argument("--asvspoof_root",   default=None,
                        help="ASVspoof 2019 root — audio training.")
    parser.add_argument("--embeddings_dir",  default=None,
                        help="Pre-extracted embeddings dir — fast multimodal training.")
    # FF++ options
    parser.add_argument("--compression",     default="c23")
    parser.add_argument("--frames_per_video", type=int, default=4)
    parser.add_argument("--manipulations",   nargs="+", default=None,
                        help="FF++ manipulations. Defaults to all five.")
    parser.add_argument("--val_split",       type=float, default=0.2)
    parser.add_argument("--face_crop",       action="store_true",
                        help="Use MTCNN face detection when loading FF++ frames.")
    # training
    parser.add_argument("--epochs",          type=int, default=20)
    parser.add_argument("--batch_size",      type=int, default=8)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--workers",         type=int, default=0)
    parser.add_argument("--checkpoint_dir",  default="checkpoints")
    parser.add_argument("--patience",        type=int, default=5)
    parser.add_argument("--mod_dropout",     type=float, default=0.15,
                        help="Per-modality dropout rate for fusion training.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    n_sources = sum([bool(args.ffpp_root), bool(args.asvspoof_root), bool(args.embeddings_dir)])
    if n_sources == 0:
        raise ValueError("Provide one of: --ffpp_root, --asvspoof_root, --embeddings_dir")
    if n_sources > 1:
        raise ValueError("Provide only one data source per run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Multimodal path: train fusion head on pre-extracted embeddings
    # ------------------------------------------------------------------
    if args.embeddings_dir:
        print(f"\nMultimodal embedding path: {args.embeddings_dir}")
        print("Loading train embeddings...")
        train_ds = MultiModalEmbeddingDataset(args.embeddings_dir, "train")
        print("Loading val embeddings...")
        val_ds   = MultiModalEmbeddingDataset(args.embeddings_dir, "val")
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

        has_img, has_aud, has_vid = train_ds.has_img, train_ds.has_aud, train_ds.has_vid
        present = [m for m, h in [("image", has_img), ("audio", has_aud), ("video", has_vid)] if h]
        print(f"Present modalities: {present}  mod_dropout={args.mod_dropout}")

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
        optimizer = AdamW(fusion.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"LR={args.lr:.2e}  Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")

        ckpt_path = os.path.join(args.checkpoint_dir, "best_fusion.pt")
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_fusion_epoch(
                train_loader, fusion, criterion, device, optimizer,
                has_img=has_img, has_aud=has_aud, has_vid=has_vid,
                mod_dropout=args.mod_dropout,
            )
            vl_loss, vl_acc = run_fusion_epoch(
                val_loader, fusion, criterion, device,
                has_img=has_img, has_aud=has_aud, has_vid=has_vid,
            )
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

        # Test eval
        test_npz = Path(args.embeddings_dir) / "image_test.npz"
        if not any((Path(args.embeddings_dir) / f"{m}_test.npz").exists()
                   for m in ("image", "audio", "video")):
            print("No test embeddings found — skipping test eval.")
        else:
            print("\nTest evaluation...")
            test_ds = MultiModalEmbeddingDataset(args.embeddings_dir, "test")
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                     num_workers=args.workers)
            best_state = torch.load(ckpt_path, map_location=device)
            fusion.load_state_dict(best_state["fusion"])
            tst_loss, tst_acc = run_fusion_epoch(
                test_loader, fusion, criterion, device,
                has_img=has_img, has_aud=has_aud, has_vid=has_vid,
            )
            print(f"Test: loss={tst_loss:.4f}  acc={tst_acc:.4f}")
        return

    # ------------------------------------------------------------------
    # Audio path: train AudioFeatureExtractor on ASVspoof
    # ------------------------------------------------------------------
    if args.asvspoof_root:
        from benchmarks.asvspoof import ASVspoofDataset

        audio_dir   = os.path.join(args.asvspoof_root, "flac")
        proto_dir   = os.path.join(args.asvspoof_root, "protocol")
        train_proto = os.path.join(proto_dir, "train.txt")
        val_proto   = os.path.join(proto_dir, "dev.txt")
        test_proto  = os.path.join(proto_dir, "eval.txt")

        train_ds = ASVspoofDataset(audio_dir, train_proto)
        val_ds   = ASVspoofDataset(audio_dir, val_proto)
        print(f"\nAudio path | Train: {len(train_ds)}  Val: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=_make_sampler([lbl for _, lbl in train_ds.samples]),
            num_workers=args.workers, pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

        aud_ext   = AudioFeatureExtractor(sample_rate=16000).to(device)
        head      = _make_head(device)
        optimizer = AdamW([
            {"params": aud_ext.lcnn.parameters(), "lr": args.lr * 0.1},
            {"params": [p for n, p in aud_ext.named_parameters() if "lcnn" not in n],
             "lr": args.lr},
            {"params": head.parameters(), "lr": args.lr},
        ], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"LR head={args.lr:.2e}  lcnn={args.lr * 0.1:.2e}")

        ckpt_path = os.path.join(args.checkpoint_dir, "best_audio.pt")
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_generic_epoch(train_loader, aud_ext, head, criterion, device, optimizer)
            vl_loss, vl_acc = run_generic_epoch(val_loader,   aud_ext, head, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                epochs_no_improve = 0
                save_checkpoint(ckpt_path, epoch, vl_acc, vl_loss, aud_ext=aud_ext, head=head)
                print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Done. Best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")

        if os.path.exists(test_proto):
            print("\nTest evaluation...")
            test_ds = ASVspoofDataset(audio_dir, test_proto)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                     num_workers=args.workers)
            best_state = torch.load(ckpt_path, map_location=device)
            aud_ext.load_state_dict(best_state["aud_extractor"])
            head.load_state_dict(best_state["head"])
            tst_loss, tst_acc = run_generic_epoch(test_loader, aud_ext, head, criterion, device)
            print(f"Test: loss={tst_loss:.4f}  acc={tst_acc:.4f}")
        return

    # ------------------------------------------------------------------
    # Image path: fine-tune XceptionNet end-to-end on FF++
    # ------------------------------------------------------------------
    manips = args.manipulations or MANIPULATIONS
    print(f"\nImage path | manipulations: {manips}  face_crop={args.face_crop}")

    base_ds_kwargs = dict(
        compression=args.compression,
        frames_per_video=args.frames_per_video,
        manipulations=manips,
        use_face_crop=args.face_crop,
    )

    # Build train/val split from FF++ "train" split
    full_ds = FaceForensicsDataset(args.ffpp_root, split="train", **base_ds_kwargs)
    all_samples = list(full_ds.samples)
    random.Random(42).shuffle(all_samples)
    n_val         = max(1, int(len(all_samples) * args.val_split))
    val_samples   = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    train_ds = FaceForensicsDataset(
        args.ffpp_root, split="train", **base_ds_kwargs,
        transform=get_train_transforms(size=299), random_frames=True,
    )
    train_ds.samples = train_samples
    val_ds = FaceForensicsDataset(
        args.ffpp_root, split="train", **base_ds_kwargs,
        transform=get_val_transforms(size=299),
    )
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

    img_ext   = ImageFeatureExtractor(pretrained=True).to(device)
    head      = _make_head(device)
    optimizer = AdamW([
        {"params": img_ext.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": [p for n, p in img_ext.named_parameters() if "backbone" not in n],
         "lr": args.lr},
        {"params": head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"LR head/srm={args.lr:.2e}  backbone={args.lr * 0.1:.2e}")

    ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0

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

    # Test eval on FF++ official test split
    print("\nTest evaluation (reloading best checkpoint)...")
    best_state = torch.load(ckpt_path, map_location=device)
    img_ext.load_state_dict(best_state["img_extractor"])
    head.load_state_dict(best_state["head"], strict=False)
    test_ds = FaceForensicsDataset(
        args.ffpp_root, split="test", **base_ds_kwargs,
        transform=get_val_transforms(size=299),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)
    tst_loss, tst_acc = run_image_epoch(test_loader, img_ext, head, criterion, device)
    print(f"Test: loss={tst_loss:.4f}  acc={tst_acc:.4f}")


if __name__ == "__main__":
    main()
