"""
train.py

Image (fine-tune XceptionNet on FF++):
    python train.py --ffpp_root /data/FF++ --modality image

Video (fine-tune R3D-18 on FF++):
    python train.py --ffpp_root /data/FF++ --modality video

Audio (train LFCC+LCNN on ASVspoof):
    python train.py --asvspoof_root /data/ASVspoof2019

Multimodal fusion (pre-extracted embeddings):
    python train.py --embeddings_dir embeddings/

Resume any run:
    python train.py ... --resume checkpoints/last.pt
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from benchmarks.ff_plusplus import MANIPULATIONS, FaceForensicsDataset
from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor
from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE, VideoFeatureExtractor
from fusion.cross_attention import MultiModalFusionClassifier
from utils.augmentations import get_train_transforms, get_val_transforms

# Video-specific transforms (112×112, ImageNet normalisation)
_VID_TRAIN_TRANSFORM = T.Compose([
    T.Resize((CLIP_SIZE, CLIP_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_VID_VAL_TRANSFORM = T.Compose([
    T.Resize((CLIP_SIZE, CLIP_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Log file: tee stdout to checkpoint_dir/train.log
# ---------------------------------------------------------------------------

class _Tee:
    """Mirrors all writes to stdout and a log file simultaneously."""
    def __init__(self, log_path, mode="a"):
        self._file = open(log_path, mode)
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# ---------------------------------------------------------------------------
# Multimodal embedding dataset (fast path: pre-extracted features)
# ---------------------------------------------------------------------------

class MultiModalEmbeddingDataset(Dataset):
    """
    Loads pre-extracted per-modality .npz files from scripts/preextract.py.
    Missing modalities are returned as zero tensors; has_* flags indicate
    which modalities have real data.
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


def _clip_grads(optimizer, max_norm=1.0):
    params = [p for pg in optimizer.param_groups for p in pg['params']]
    nn.utils.clip_grad_norm_(params, max_norm)


def run_image_epoch(loader, img_ext, head, criterion, device, optimizer=None, clip_grad=1.0):
    training = optimizer is not None
    img_ext.train(training)
    head.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for frames, labels in loader:
            labels = labels.float().to(device)
            frames = frames.to(device)
            if frames.ndim == 5:             # (B, T, C, H, W) — multi-frame
                B, T, C, H, W = frames.shape
                emb = img_ext(frames.view(B * T, C, H, W)).view(B, T, -1).mean(1)
            else:                            # (B, C, H, W) — single image
                emb = img_ext(frames)
            preds = head(emb).squeeze(1)
            loss = criterion(preds, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                _clip_grads(optimizer, clip_grad)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0).float() == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


def run_video_epoch(loader, vid_ext, head, criterion, device, optimizer=None, clip_grad=1.0):
    training = optimizer is not None
    vid_ext.train(training)
    head.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for clips, labels in loader:
            labels = labels.float().to(device)
            emb = vid_ext(clips.to(device))        # (B, T, C, H, W) → (B, 512)
            preds = head(emb).squeeze(1)
            loss = criterion(preds, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                _clip_grads(optimizer, clip_grad)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0).float() == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


def run_generic_epoch(loader, extractor, head, criterion, device, optimizer=None, clip_grad=1.0):
    """Generic epoch for single-input extractors (audio)."""
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
                _clip_grads(optimizer, clip_grad)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0).float() == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


def run_fusion_epoch(loader, fusion, criterion, device, optimizer=None,
                     has_img=True, has_aud=True, has_vid=True, mod_dropout=0.15,
                     clip_grad=1.0):
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
                _clip_grads(optimizer, clip_grad)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0).float() == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


def save_checkpoint(path, epoch, val_acc, val_loss,
                    img_ext=None, aud_ext=None, vid_ext=None,
                    fusion=None, head=None,
                    optimizer=None, scheduler=None):
    torch.save(
        {
            "epoch":         epoch,
            "val_acc":       round(val_acc, 6),
            "val_loss":      round(val_loss, 6),
            "img_extractor": img_ext.state_dict() if img_ext  is not None else {},
            "aud_extractor": aud_ext.state_dict() if aud_ext  is not None else {},
            "vid_extractor": vid_ext.state_dict() if vid_ext  is not None else {},
            "fusion":        fusion.state_dict()  if fusion    is not None else {},
            "head":          head.state_dict()    if head      is not None else {},
            "optimizer":     optimizer.state_dict() if optimizer is not None else {},
            "scheduler":     scheduler.state_dict() if scheduler is not None else {},
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


def _make_scheduler(optimizer, args):
    """Linear warmup then cosine annealing. Falls back to plain cosine if warmup=0."""
    if args.warmup_epochs > 0 and args.epochs > args.warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine],
                            milestones=[args.warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=args.epochs)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    # data sources
    parser.add_argument("--ffpp_root",        default=None)
    parser.add_argument("--asvspoof_root",    default=None)
    parser.add_argument("--embeddings_dir",   default=None)
    # FF++ options
    parser.add_argument("--modality",         choices=["image", "video"], default="image",
                        help="Which modality to train when using --ffpp_root.")
    parser.add_argument("--compression",      default="c23")
    parser.add_argument("--frames_per_video", type=int, default=4)
    parser.add_argument("--manipulations",    nargs="+", default=None)
    parser.add_argument("--val_split",        type=float, default=0.2)
    parser.add_argument("--face_crop",        action="store_true")
    parser.add_argument("--kaggle_root",      default=None,
                        help="Path to 140k Real and Fake Faces dataset (image-only training).")
    # training
    parser.add_argument("--fake_weight",      type=float, default=1.0,
                        help="BCEWithLogitsLoss pos_weight for fake samples (>1.0 upweights fakes).")
    parser.add_argument("--epochs",           type=int, default=30)
    parser.add_argument("--batch_size",       type=int, default=8)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--workers",          type=int, default=0)
    parser.add_argument("--checkpoint_dir",   default="checkpoints")
    parser.add_argument("--patience",         type=int, default=7)
    parser.add_argument("--warmup_epochs",    type=int, default=3,
                        help="Linear warmup epochs before cosine annealing kicks in.")
    parser.add_argument("--save_every",       type=int, default=5,
                        help="Save a periodic checkpoint every N epochs (0 to disable).")
    parser.add_argument("--clip_grad",        type=float, default=1.0,
                        help="Max gradient norm for clipping (0 to disable).")
    parser.add_argument("--mod_dropout",      type=float, default=0.15)
    parser.add_argument("--resume",           default=None,
                        help="Checkpoint path to resume training from.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    n_sources = sum([bool(args.ffpp_root), bool(args.asvspoof_root),
                     bool(args.embeddings_dir), bool(args.kaggle_root)])
    if n_sources == 0:
        raise ValueError("Provide one of: --ffpp_root, --asvspoof_root, --embeddings_dir")
    if n_sources > 1:
        raise ValueError("Provide only one data source per run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Set up log file tee
    log_path = Path(args.checkpoint_dir) / "train.log"
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        print(f"Device: {device}")
        pos_w = torch.tensor([args.fake_weight], device=device) if args.fake_weight != 1.0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        # ---------------------------------------------------------------
        # Multimodal fusion path
        # ---------------------------------------------------------------
        if args.embeddings_dir:
            print(f"\nMultimodal path: {args.embeddings_dir}")
            print("Loading train embeddings...")
            train_ds = MultiModalEmbeddingDataset(args.embeddings_dir, "train")
            print("Loading val embeddings...")
            val_ds   = MultiModalEmbeddingDataset(args.embeddings_dir, "val")
            print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

            has_img, has_aud, has_vid = train_ds.has_img, train_ds.has_aud, train_ds.has_vid
            present = [m for m, h in [("image", has_img), ("audio", has_aud), ("video", has_vid)] if h]
            print(f"Present modalities: {present}  mod_dropout={args.mod_dropout}")

            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=_make_sampler(train_ds.labels.tolist()),
                                      num_workers=args.workers, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)

            fusion    = MultiModalFusionClassifier().to(device)
            optimizer = AdamW(fusion.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = _make_scheduler(optimizer, args)

            start_epoch, best_val_acc = 1, 0.0
            if args.resume:
                ckpt = torch.load(args.resume, map_location=device)
                fusion.load_state_dict(ckpt.get("fusion", {}), strict=False)
                if ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
                if ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch  = ckpt.get("epoch", 0) + 1
                best_val_acc = ckpt.get("val_acc", 0.0)
                print(f"Resumed from {args.resume} (epoch {start_epoch-1}, val_acc={best_val_acc:.4f})")

            ckpt_path  = os.path.join(args.checkpoint_dir, "best_fusion.pt")
            last_path  = os.path.join(args.checkpoint_dir, "last_fusion.pt")
            print(f"LR={args.lr:.2e}  warmup={args.warmup_epochs}ep  patience={args.patience}")
            print(f"Params: {sum(p.numel() for p in fusion.parameters()):,}")

            epochs_no_improve = 0
            for epoch in range(start_epoch, args.epochs + 1):
                tr_loss, tr_acc = run_fusion_epoch(
                    train_loader, fusion, criterion, device, optimizer,
                    has_img=has_img, has_aud=has_aud, has_vid=has_vid,
                    mod_dropout=args.mod_dropout, clip_grad=args.clip_grad)
                vl_loss, vl_acc = run_fusion_epoch(
                    val_loader, fusion, criterion, device,
                    has_img=has_img, has_aud=has_aud, has_vid=has_vid)
                scheduler.step()
                print(f"Epoch {epoch:3d}/{args.epochs}  "
                      f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                      f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")
                save_checkpoint(last_path, epoch, vl_acc, vl_loss,
                                fusion=fusion, optimizer=optimizer, scheduler=scheduler)
                if args.save_every > 0 and epoch % args.save_every == 0:
                    periodic = os.path.join(args.checkpoint_dir, f"fusion_epoch{epoch:03d}.pt")
                    save_checkpoint(periodic, epoch, vl_acc, vl_loss,
                                    fusion=fusion, optimizer=optimizer, scheduler=scheduler)
                    print(f"  Periodic save -> {periodic}")
                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc; epochs_no_improve = 0
                    save_checkpoint(ckpt_path, epoch, vl_acc, vl_loss,
                                    fusion=fusion, optimizer=optimizer, scheduler=scheduler)
                    print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.patience:
                        print(f"Early stopping at epoch {epoch}"); break

            print(f"Done. Best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")

            # Test eval
            if any((Path(args.embeddings_dir) / f"{m}_test.npz").exists()
                   for m in ("image", "audio", "video")):
                print("\nTest evaluation...")
                test_ds = MultiModalEmbeddingDataset(args.embeddings_dir, "test")
                test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                         num_workers=args.workers)
                best_state = torch.load(ckpt_path, map_location=device)
                fusion.load_state_dict(best_state["fusion"])
                tst_loss, tst_acc = run_fusion_epoch(
                    test_loader, fusion, criterion, device,
                    has_img=has_img, has_aud=has_aud, has_vid=has_vid)
                print(f"Test: loss={tst_loss:.4f}  acc={tst_acc:.4f}")
            return

        # ---------------------------------------------------------------
        # Audio path: ASVspoof
        # ---------------------------------------------------------------
        if args.asvspoof_root:
            from benchmarks.asvspoof import ASVspoofDataset
            audio_dir   = os.path.join(args.asvspoof_root, "flac")
            proto_dir   = os.path.join(args.asvspoof_root, "protocol")
            train_ds = ASVspoofDataset(audio_dir, os.path.join(proto_dir, "train.txt"))
            val_ds   = ASVspoofDataset(audio_dir, os.path.join(proto_dir, "dev.txt"))
            print(f"\nAudio path | Train: {len(train_ds)}  Val: {len(val_ds)}")

            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=_make_sampler([lbl for _, lbl in train_ds.samples]),
                                      num_workers=args.workers, pin_memory=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)

            aud_ext   = AudioFeatureExtractor(sample_rate=16000).to(device)
            head      = _make_head(device)
            optimizer = AdamW([
                {"params": aud_ext.lcnn.parameters(),
                 "lr": args.lr * 0.1},
                {"params": [p for n, p in aud_ext.named_parameters() if "lcnn" not in n],
                 "lr": args.lr},
                {"params": head.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = _make_scheduler(optimizer, args)

            start_epoch, best_val_acc = 1, 0.0
            if args.resume:
                ckpt = torch.load(args.resume, map_location=device)
                aud_ext.load_state_dict(ckpt.get("aud_extractor", {}), strict=False)
                head.load_state_dict(ckpt.get("head", {}), strict=False)
                if ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
                if ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch  = ckpt.get("epoch", 0) + 1
                best_val_acc = ckpt.get("val_acc", 0.0)
                print(f"Resumed from {args.resume} (epoch {start_epoch-1}, val_acc={best_val_acc:.4f})")

            ckpt_path = os.path.join(args.checkpoint_dir, "best_audio.pt")
            last_path = os.path.join(args.checkpoint_dir, "last_audio.pt")
            print(f"LR head={args.lr:.2e}  lcnn={args.lr*0.1:.2e}  warmup={args.warmup_epochs}ep")

            epochs_no_improve = 0
            for epoch in range(start_epoch, args.epochs + 1):
                tr_loss, tr_acc = run_generic_epoch(train_loader, aud_ext, head, criterion, device,
                                                    optimizer, clip_grad=args.clip_grad)
                vl_loss, vl_acc = run_generic_epoch(val_loader,   aud_ext, head, criterion, device)
                scheduler.step()
                print(f"Epoch {epoch:3d}/{args.epochs}  "
                      f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                      f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")
                save_checkpoint(last_path, epoch, vl_acc, vl_loss,
                                aud_ext=aud_ext, head=head,
                                optimizer=optimizer, scheduler=scheduler)
                if args.save_every > 0 and epoch % args.save_every == 0:
                    periodic = os.path.join(args.checkpoint_dir, f"audio_epoch{epoch:03d}.pt")
                    save_checkpoint(periodic, epoch, vl_acc, vl_loss,
                                    aud_ext=aud_ext, head=head,
                                    optimizer=optimizer, scheduler=scheduler)
                    print(f"  Periodic save -> {periodic}")
                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc; epochs_no_improve = 0
                    save_checkpoint(ckpt_path, epoch, vl_acc, vl_loss,
                                    aud_ext=aud_ext, head=head,
                                    optimizer=optimizer, scheduler=scheduler)
                    print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.patience:
                        print(f"Early stopping at epoch {epoch}"); break

            print(f"Done. Best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")

            test_proto = os.path.join(proto_dir, "eval.txt")
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

        # ---------------------------------------------------------------
        # Kaggle 140k Real and Fake Faces path (image-only)
        # ---------------------------------------------------------------
        if args.kaggle_root:
            from benchmarks.kaggle_faces import KaggleFacesDataset
            train_ds = KaggleFacesDataset(args.kaggle_root, split="train",
                                          transform=get_train_transforms(size=299))
            val_ds   = KaggleFacesDataset(args.kaggle_root, split="val",
                                          transform=get_val_transforms(size=299))
            print(f"\nKaggle path | Train: {len(train_ds)}  Val: {len(val_ds)}")

            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=_make_sampler([lbl for _, lbl in train_ds.samples]),
                                      num_workers=args.workers, pin_memory=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)

            img_ext   = ImageFeatureExtractor(pretrained=True).to(device)
            head      = _make_head(device)
            optimizer = AdamW([
                {"params": img_ext.backbone.parameters(), "lr": args.lr * 0.1},
                {"params": [p for n, p in img_ext.named_parameters() if "backbone" not in n],
                 "lr": args.lr},
                {"params": head.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = _make_scheduler(optimizer, args)

            start_epoch, best_val_acc = 1, 0.0
            if args.resume:
                ckpt = torch.load(args.resume, map_location=device)
                img_ext.load_state_dict(ckpt.get("img_extractor", {}), strict=False)
                head.load_state_dict(ckpt.get("head", {}), strict=False)
                if ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
                if ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch  = ckpt.get("epoch", 0) + 1
                best_val_acc = ckpt.get("val_acc", 0.0)
                print(f"Resumed from {args.resume} (epoch {start_epoch-1}, val_acc={best_val_acc:.4f})")

            ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
            last_path = os.path.join(args.checkpoint_dir, "last.pt")
            print(f"LR head/srm={args.lr:.2e}  backbone={args.lr*0.1:.2e}  warmup={args.warmup_epochs}ep")

            epochs_no_improve = 0
            for epoch in range(start_epoch, args.epochs + 1):
                tr_loss, tr_acc = run_image_epoch(train_loader, img_ext, head, criterion, device,
                                                  optimizer, clip_grad=args.clip_grad)
                vl_loss, vl_acc = run_image_epoch(val_loader,   img_ext, head, criterion, device)
                scheduler.step()
                print(f"Epoch {epoch:3d}/{args.epochs}  "
                      f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                      f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")
                save_checkpoint(last_path, epoch, vl_acc, vl_loss,
                                img_ext=img_ext, head=head,
                                optimizer=optimizer, scheduler=scheduler)
                if args.save_every > 0 and epoch % args.save_every == 0:
                    periodic = os.path.join(args.checkpoint_dir, f"epoch{epoch:03d}.pt")
                    save_checkpoint(periodic, epoch, vl_acc, vl_loss,
                                    img_ext=img_ext, head=head,
                                    optimizer=optimizer, scheduler=scheduler)
                    print(f"  Periodic save -> {periodic}")
                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc; epochs_no_improve = 0
                    save_checkpoint(ckpt_path, epoch, vl_acc, vl_loss,
                                    img_ext=img_ext, head=head,
                                    optimizer=optimizer, scheduler=scheduler)
                    print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.patience:
                        print(f"Early stopping at epoch {epoch}"); break

            print(f"Done. Best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")

            test_ds = KaggleFacesDataset(args.kaggle_root, split="test",
                                          transform=get_val_transforms(size=299))
            if len(test_ds) > 0:
                print("\nTest evaluation...")
                test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)
                best_state = torch.load(ckpt_path, map_location=device)
                img_ext.load_state_dict(best_state["img_extractor"], strict=False)
                head.load_state_dict(best_state["head"], strict=False)
                tst_loss, tst_acc = run_image_epoch(test_loader, img_ext, head, criterion, device)
                print(f"Test: loss={tst_loss:.4f}  acc={tst_acc:.4f}")
            return

        # ---------------------------------------------------------------
        # FF++ path: image or video
        # ---------------------------------------------------------------
        manips = args.manipulations or MANIPULATIONS
        is_video = args.modality == "video"
        fpv = CLIP_FRAMES if is_video else args.frames_per_video
        print(f"\n{'Video' if is_video else 'Image'} path | manipulations: {manips}  "
              f"face_crop={args.face_crop}")

        base_kwargs = dict(
            compression=args.compression,
            frames_per_video=fpv,
            manipulations=manips,
            use_face_crop=args.face_crop,
        )
        train_tfm = _VID_TRAIN_TRANSFORM if is_video else get_train_transforms(size=299)
        val_tfm   = _VID_VAL_TRANSFORM   if is_video else get_val_transforms(size=299)

        full_ds = FaceForensicsDataset(args.ffpp_root, split="train", **base_kwargs)
        all_samples = list(full_ds.samples)
        random.Random(42).shuffle(all_samples)
        n_val         = max(1, int(len(all_samples) * args.val_split))
        val_samples   = all_samples[:n_val]
        train_samples = all_samples[n_val:]

        train_ds = FaceForensicsDataset(args.ffpp_root, split="train", **base_kwargs,
                                        transform=train_tfm,
                                        random_frames=(not is_video))
        train_ds.samples = train_samples
        val_ds = FaceForensicsDataset(args.ffpp_root, split="train", **base_kwargs,
                                      transform=val_tfm)
        val_ds.samples = val_samples
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=_make_sampler([lbl for _, lbl in train_samples]),
                                  num_workers=args.workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

        if is_video:
            vid_ext   = VideoFeatureExtractor(pretrained=True).to(device)
            head      = _make_head(device)
            optimizer = AdamW([
                {"params": [p for n, p in vid_ext.named_parameters() if "proj" not in n],
                 "lr": args.lr * 0.1},
                {"params": vid_ext.proj.parameters(), "lr": args.lr},
                {"params": head.parameters(),          "lr": args.lr},
            ], weight_decay=1e-4)
        else:
            img_ext   = ImageFeatureExtractor(pretrained=True).to(device)
            head      = _make_head(device)
            optimizer = AdamW([
                {"params": img_ext.backbone.parameters(), "lr": args.lr * 0.1},
                {"params": [p for n, p in img_ext.named_parameters() if "backbone" not in n],
                 "lr": args.lr},
                {"params": head.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)

        scheduler = _make_scheduler(optimizer, args)

        start_epoch, best_val_acc = 1, 0.0
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            key = "vid_extractor" if is_video else "img_extractor"
            model = vid_ext if is_video else img_ext
            model.load_state_dict(ckpt.get(key, {}), strict=False)
            head.load_state_dict(ckpt.get("head", {}), strict=False)
            if ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch  = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("val_acc", 0.0)
            print(f"Resumed from {args.resume} (epoch {start_epoch-1}, val_acc={best_val_acc:.4f})")

        if is_video:
            print(f"LR proj/head={args.lr:.2e}  R3D-18 layers={args.lr*0.1:.2e}  warmup={args.warmup_epochs}ep")
        else:
            print(f"LR head/srm={args.lr:.2e}  backbone={args.lr*0.1:.2e}  warmup={args.warmup_epochs}ep")

        ckpt_name = "best_video.pt" if is_video else "best.pt"
        last_name = "last_video.pt" if is_video else "last.pt"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        last_path = os.path.join(args.checkpoint_dir, last_name)
        pfx       = "video" if is_video else "epoch"
        run_epoch = run_video_epoch if is_video else run_image_epoch
        ext       = vid_ext if is_video else img_ext

        epochs_no_improve = 0
        for epoch in range(start_epoch, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(train_loader, ext, head, criterion, device,
                                        optimizer, clip_grad=args.clip_grad)
            vl_loss, vl_acc = run_epoch(val_loader,   ext, head, criterion, device)
            scheduler.step()
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")
            save_kw = dict(vid_ext=ext, head=head) if is_video else dict(img_ext=ext, head=head)
            save_checkpoint(last_path, epoch, vl_acc, vl_loss,
                            optimizer=optimizer, scheduler=scheduler, **save_kw)
            if args.save_every > 0 and epoch % args.save_every == 0:
                periodic = os.path.join(args.checkpoint_dir, f"{pfx}{epoch:03d}.pt")
                save_checkpoint(periodic, epoch, vl_acc, vl_loss,
                                optimizer=optimizer, scheduler=scheduler, **save_kw)
                print(f"  Periodic save -> {periodic}")
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc; epochs_no_improve = 0
                save_checkpoint(ckpt_path, epoch, vl_acc, vl_loss,
                                optimizer=optimizer, scheduler=scheduler, **save_kw)
                print(f"  Saved {ckpt_path} (val_acc={vl_acc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}"); break

        print(f"Done. Best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")

        # Test eval
        print("\nTest evaluation (reloading best checkpoint)...")
        best_state = torch.load(ckpt_path, map_location=device)
        key = "vid_extractor" if is_video else "img_extractor"
        ext.load_state_dict(best_state[key], strict=False)
        head.load_state_dict(best_state["head"], strict=False)
        test_ds = FaceForensicsDataset(args.ffpp_root, split="test", **base_kwargs,
                                       transform=val_tfm)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers)
        tst_loss, tst_acc = run_epoch(test_loader, ext, head, criterion, device)
        print(f"Test: loss={tst_loss:.4f}  acc={tst_acc:.4f}")

    finally:
        tee.close()


if __name__ == "__main__":
    main()
