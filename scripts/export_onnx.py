"""
Export trained models to ONNX for deployment.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output_dir onnx/

Exports:
    onnx/image_extractor.onnx   (B, 3, 299, 299)  → (B, 512)
    onnx/audio_extractor.onnx   (B, 64000)         → (B, 512)   [may skip if STFT fails]
    onnx/video_extractor.onnx   (B, 16, 3, 112, 112) → (B, 512)
    onnx/head.onnx              (B, 512)            → (B, 1)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor
from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE, VideoFeatureExtractor


def try_export(model, dummy, path, input_names, output_names, dynamic_axes, opset=17):
    try:
        torch.onnx.export(
            model, dummy, str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
        size_mb = path.stat().st_size / 1e6
        print(f"  OK   {path.name}  ({size_mb:.1f} MB)")

        # Quick validation with onnxruntime if available
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(path))
            dummy_np = {n: d.cpu().numpy() for n, d in zip(input_names,
                        dummy if isinstance(dummy, (list, tuple)) else [dummy])}
            out = sess.run(None, dummy_np)
            print(f"       onnxruntime check passed  output shape: {out[0].shape}")
        except ImportError:
            print(f"       (install onnxruntime to validate: pip install onnxruntime)")
        except Exception as e:
            print(f"       onnxruntime validation failed: {e}")

    except Exception as e:
        print(f"  SKIP {path.name}  — {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output_dir",  default="onnx")
    parser.add_argument("--opset",       type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cpu")  # ONNX export on CPU is safest
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location=device)

    # -----------------------------------------------------------------------
    # Image extractor
    # -----------------------------------------------------------------------
    img_ext = ImageFeatureExtractor(pretrained=False)
    img_ext.load_state_dict(state.get("img_extractor", {}), strict=False)
    img_ext.eval()
    print("\nExporting image extractor...")
    try_export(
        img_ext,
        torch.zeros(1, 3, 299, 299),
        out / "image_extractor.onnx",
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset=args.opset,
    )

    # -----------------------------------------------------------------------
    # Audio extractor  (STFT may not be ONNX-compatible on all runtimes)
    # -----------------------------------------------------------------------
    aud_ext = AudioFeatureExtractor(sample_rate=16000)
    aud_ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
    aud_ext.eval()
    print("\nExporting audio extractor...")
    try_export(
        aud_ext,
        torch.zeros(1, 64000),
        out / "audio_extractor.onnx",
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={"waveform": {0: "batch"}, "embedding": {0: "batch"}},
        opset=args.opset,
    )

    # -----------------------------------------------------------------------
    # Video extractor
    # -----------------------------------------------------------------------
    vid_ext = VideoFeatureExtractor(pretrained=False)
    vid_ext.load_state_dict(state.get("vid_extractor", {}), strict=False)
    vid_ext.eval()
    print("\nExporting video extractor...")
    try_export(
        vid_ext,
        torch.zeros(1, CLIP_FRAMES, 3, CLIP_SIZE, CLIP_SIZE),
        out / "video_extractor.onnx",
        input_names=["clip"],
        output_names=["embedding"],
        dynamic_axes={"clip": {0: "batch"}, "embedding": {0: "batch"}},
        opset=args.opset,
    )

    # -----------------------------------------------------------------------
    # Head classifier
    # -----------------------------------------------------------------------
    head = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    head.load_state_dict(state.get("head", {}), strict=False)
    head.eval()
    print("\nExporting head classifier...")
    try_export(
        head,
        torch.zeros(1, 512),
        out / "head.onnx",
        input_names=["embedding"],
        output_names=["fake_prob"],
        dynamic_axes={"embedding": {0: "batch"}, "fake_prob": {0: "batch"}},
        opset=args.opset,
    )

    print(f"\nAll exports attempted → {out}/")


if __name__ == "__main__":
    main()
