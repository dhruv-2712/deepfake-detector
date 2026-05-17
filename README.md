---
title: Deepfake Detector
emoji: 🎭
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# Deepfake Detector

A multimodal deepfake detection system that analyses images, audio, and video using three specialist neural networks fused via cross-attention.

**83.3% balanced accuracy · 76.5% AUC** on FaceForensics++ c40 (held-out test set)

[Live Demo](https://huggingface.co/spaces/dhruv2712/deepfake-detector) · [Model Weights](https://huggingface.co/dhruv2712/deepfake-detector-weights)

---

## Architecture

Three extractors each output a 512-dimensional embedding, which a cross-attention fusion classifier combines into a single fake probability:

| Modality | Model | What it detects |
|---|---|---|
| Image | XceptionNet + SRM noise filters | GAN texture artifacts, compression anomalies |
| Audio | LFCC + LCNN | Voice cloning artifacts in frequency patterns |
| Video | R3D-18 (3D CNN) | Unnatural motion, temporal inconsistency |

The fusion classifier uses multi-head cross-attention so it can operate on any subset of modalities — image-only, audio-only, video-only, or any combination.

---

## Results

Trained on **FaceForensics++ c40** (heavy compression — the hardest variant):

| Metric | Value |
|---|---|
| Balanced Accuracy (val) | 83.3% |
| AUC (held-out test) | 76.5% |
| Average Precision | 81.5% |
| Equal Error Rate | 30.2% |

Covers all five FF++ manipulation types: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter.

> **Note:** The model detects face-swap and face-reenactment fakes (FF++ style). It was not trained on fully GAN-synthesized faces (e.g. StyleGAN).

---

## Project Structure

```
deepfake-detector/
│
├── detectors/                  # Feature extractors (one per modality)
│   ├── image/
│   │   └── extractor.py        # XceptionNet + SRM residual noise filters
│   ├── audio/
│   │   └── extractor.py        # LFCC features + LCNN classifier
│   └── video/
│       └── extractor.py        # R3D-18 3D CNN temporal extractor
│
├── fusion/
│   └── cross_attention.py      # Multi-head cross-attention + MLP fusion head
│
├── benchmarks/                 # Dataset loaders
│   ├── ff_plusplus.py          # FaceForensics++ (video-based)
│   ├── kaggle_faces.py         # 140k Real and Fake Faces (JPEG-based)
│   └── asvspoof.py             # ASVspoof 2019 (audio)
│
├── utils/
│   ├── face_align.py           # MTCNN face detection and alignment
│   ├── gradcam.py              # Grad-CAM heatmap over XceptionNet
│   └── augmentations.py        # JPEG, noise, and flip augmentations
│
├── scripts/
│   ├── preextract.py           # Pre-extract embeddings to disk
│   └── export_onnx.py          # Export models to ONNX
│
├── api/
│   └── main.py                 # FastAPI REST endpoints
│
├── train.py                    # Training script
├── eval.py                     # Evaluation (AUC, AP, EER)
├── demo.py                     # Gradio web demo
├── extract_frames.py           # One-time FF++ video → JPEG extraction
└── app.py                      # HuggingFace Spaces entry point
```

---

## Setup

```bash
pip install -r requirements.txt
```

PyTorch with CUDA (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

### Step 1 — Extract frames from FF++ videos (one-time, much faster training)

```bash
python extract_frames.py --ffpp_root data/ffpp --out data/ffpp_frames --compression c40
```

This converts MP4 videos to JPEG frames so each training epoch takes minutes instead of hours.

### Step 2 — Train the image detector

```bash
python train.py \
  --kaggle_root data/ffpp_frames \
  --checkpoint_dir checkpoints/ffpp \
  --fake_weight 5.0 \
  --workers 4
```

`--fake_weight 5.0` compensates for the 5:1 fake:real imbalance in FF++ (5 manipulation types × 720 videos vs 720 real videos).

### Other modalities

```bash
# Audio (ASVspoof 2019)
python train.py --asvspoof_root data/ASVspoof2019 --checkpoint_dir checkpoints/audio

# Video (FF++ raw videos, slower)
python train.py --ffpp_root data/ffpp --compression c40 --modality video --checkpoint_dir checkpoints/video
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 30 | Training epochs |
| `--batch_size` | 8 | Batch size (keep low for 6GB VRAM) |
| `--fake_weight` | 1.0 | BCEWithLogitsLoss pos_weight for class imbalance |
| `--patience` | 10 | Early stopping patience |
| `--workers` | 0 | DataLoader workers (set 4+ on Linux/Mac) |

---

## Evaluation

```bash
python eval.py \
  --checkpoint checkpoints/ffpp/best.pt \
  --kaggle_root data/ffpp_frames
```

Reports AUC, Average Precision, and EER on the held-out test split.

---

## Demo

```bash
python demo.py --checkpoint checkpoints/ffpp/best.pt
```

Opens at `http://localhost:7860`. Upload an image, audio clip, or video to get:
- Fake probability score (0–1)
- Face detection bounding boxes
- Grad-CAM heatmap showing which regions triggered the detection
- DCT frequency analysis score

---

## REST API

```bash
python api/main.py --checkpoint checkpoints/ffpp/best.pt
```

| Endpoint | Input | Returns |
|---|---|---|
| `POST /detect/image` | image file | `{"is_fake": bool, "confidence": float}` |
| `POST /detect/audio` | audio file | `{"is_fake": bool, "confidence": float}` |
| `POST /detect/video` | video file | `{"is_fake": bool, "confidence": float, "modalities_used": [...]}` |

Interactive docs at `http://localhost:8000/docs`.

---

## How Training Works

- **Balanced accuracy** `(TPR + TNR) / 2` is used as the early-stopping metric instead of raw accuracy, which would be gamed by always predicting the majority class in a 5:1 imbalanced dataset.
- **Linear warmup** over 3 epochs followed by cosine annealing prevents early divergence.
- **Gradient clipping** (`max_norm=1.0`) stabilises training on XceptionNet's large backbone.
- **WeightedRandomSampler** oversamples the minority (real) class within each batch.
