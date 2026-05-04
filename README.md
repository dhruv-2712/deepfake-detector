# Deepfake Detector

A multimodal deepfake detection system that analyses images, audio, and video to classify media as real or AI-generated.

## How it works

Three specialist models each extract a 512-dimensional fingerprint from their respective input:

| Modality | Model | What it detects |
|---|---|---|
| Image | XceptionNet + SRM noise filters | GAN texture artifacts, camera noise anomalies |
| Audio | LFCC + LCNN | Voice cloning artifacts in frequency patterns |
| Video | R3D-18 (3D CNN) | Unnatural motion, temporal inconsistency |

A cross-attention fusion classifier combines whichever modalities are available and outputs a single **fake probability (0–1)**.

## Project structure

```
detectors/
  image/extractor.py       # XceptionNet + SRM + DCT analysis
  audio/extractor.py       # LFCC feature extraction + LCNN
  video/extractor.py       # R3D-18 temporal feature extraction
fusion/
  cross_attention.py       # Multi-head attention fusion + MLP classifier
benchmarks/
  ff_plusplus.py           # FaceForensics++ dataset loader
  asvspoof.py              # ASVspoof 2019 dataset loader
  kaggle_faces.py          # 140k Real and Fake Faces dataset loader
utils/
  face_align.py            # MTCNN face detection and cropping
  gradcam.py               # Grad-CAM heatmap visualization
  augmentations.py         # Training augmentations (JPEG, noise, flips)
scripts/
  preextract.py            # Pre-extract embeddings to disk for faster training
  export_onnx.py           # Export trained models to ONNX
api/main.py                # FastAPI REST endpoints
demo.py                    # Gradio web UI
train.py                   # Training script
eval.py                    # Evaluation script
```

## Setup

```bash
pip install -r requirements.txt
```

PyTorch with CUDA (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Running the demo

```bash
python demo.py                                   # random weights
python demo.py --checkpoint checkpoints/best.pt  # trained weights
```

Opens at `http://localhost:7860`. Supports image, audio, and video input with face detection, Grad-CAM heatmap, and fake probability score.

## Training

### 140k Real and Fake Faces (easiest, image-only)

Download from Kaggle: `xhlulu/140k-real-and-fake-faces`

```bash
python train.py --modality image \
                --kaggle_root /path/to/kaggle-faces \
                --workers 2
```

### FaceForensics++ (image or video)

```bash
python train.py --modality image --ffpp_root /path/to/FaceForensics++
python train.py --modality video --ffpp_root /path/to/FaceForensics++
```

### ASVspoof 2019 (audio)

```bash
python train.py --modality audio --asvspoof_root /path/to/ASVspoof2019
```

Best checkpoint is saved automatically to `checkpoints/best.pt`.

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--fake_weight` | 2.0 | Loss weight for fake samples |
| `--workers` | 4 | DataLoader workers (use 0 on Windows CPU) |

## Evaluation

```bash
python eval.py --checkpoint checkpoints/best.pt \
               --ffpp_root /path/to/FaceForensics++ \
               --modality image
```

Reports AUC, Average Precision, and EER per manipulation type (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter).

## REST API

```bash
uvicorn api.main:app --reload
```

| Endpoint | Input |
|---|---|
| `POST /detect/image` | image file |
| `POST /detect/audio` | audio file |
| `POST /detect/video` | video file |

Returns `{"is_fake": true/false, "confidence": 0.91, "modalities_used": ["image"]}`.

Interactive docs at `http://localhost:8000/docs`.

## Faster training with pre-extracted embeddings

```bash
# Extract once
python scripts/preextract.py --checkpoint checkpoints/best.pt \
                              --ffpp_root /data/FaceForensics++ \
                              --modality image \
                              --output_dir embeddings/

# Train fusion head only (10-50x faster)
python train.py --modality image --embeddings_dir embeddings/
```

## Export to ONNX

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output_dir onnx/
```

Exports all five components (image extractor, audio extractor, video extractor, head classifier, fusion classifier) separately.
