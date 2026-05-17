"""
Gradio demo for the deepfake detector.

    python demo.py                              # untrained weights, local
    python demo.py --checkpoint checkpoints/best.pt
    python demo.py --share                      # public URL via Gradio tunnel
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor, dct_peak_score
from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE, VideoFeatureExtractor
from fusion.cross_attention import MultiModalFusionClassifier
from utils.face_align import FaceDetector
from utils.gradcam import DeepfakeGradCAM

# ---------------------------------------------------------------------------
# Models (loaded at module level so Gradio workers share them)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_ext  = ImageFeatureExtractor(pretrained=True).to(device).eval()
aud_ext  = AudioFeatureExtractor(sample_rate=16000).to(device).eval()
vid_ext  = VideoFeatureExtractor(pretrained=True).to(device).eval()
fusion   = MultiModalFusionClassifier().to(device).eval()
head     = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 1),
).to(device).eval()
face_det = FaceDetector(device=device)

# Track whether trained fusion weights were loaded; built lazily after checkpoint load.
_fusion_loaded = False
gradcam: DeepfakeGradCAM | None = None


class _FusionImageScorer(nn.Module):
    """Wraps fusion so GradCAM can call it with a single image embedding tensor."""
    def __init__(self, fusion_model: MultiModalFusionClassifier):
        super().__init__()
        self.fusion = fusion_model

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fusion(img_emb=emb)


def _build_gradcam():
    """Build GradCAM against whichever scorer actually has trained weights."""
    global gradcam
    scorer = _FusionImageScorer(fusion) if _fusion_loaded else head
    gradcam = DeepfakeGradCAM(img_ext, scorer)


def _predict_proba(img_emb=None, aud_emb=None, vid_emb=None) -> float:
    """Returns fake probability in [0, 1] using whichever classifier was trained."""
    with torch.no_grad():
        if _fusion_loaded:
            logits = fusion(img_emb=img_emb, aud_emb=aud_emb, vid_emb=vid_emb)
        else:
            emb = img_emb if img_emb is not None else (vid_emb if vid_emb is not None else aud_emb)
            logits = head(emb)
    return torch.sigmoid(logits).squeeze().item()

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
_VID_TRANSFORM = transforms.Compose([
    transforms.Resize((CLIP_SIZE, CLIP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_checkpoint(path: str):
    global _fusion_loaded
    state = torch.load(path, map_location=device)
    img_ext.load_state_dict(state.get("img_extractor", {}), strict=False)
    aud_ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
    vid_ext.load_state_dict(state.get("vid_extractor", {}), strict=False)
    head_state = state.get("head", {})
    if head_state:
        head.load_state_dict(head_state)
    fusion_state = state.get("fusion", {})
    if fusion_state:
        fusion.load_state_dict(fusion_state)
        _fusion_loaded = True
    val_info = f"  val_acc={state['val_acc']:.4f}" if "val_acc" in state else ""
    classifier = "fusion" if _fusion_loaded else "head"
    print(f"Loaded checkpoint: {path}  (classifier={classifier}){val_info}")


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_image(image: Image.Image):
    if image is None:
        return "Upload an image to begin.", 0.0, "", None, None

    image     = image.convert("RGB")
    annotated = face_det.draw_boxes(image)

    # Model was trained on plain resized images — match that preprocessing exactly
    tensor = _IMG_TRANSFORM(image).unsqueeze(0).to(device)

    # Grad-CAM needs gradients — run before the no_grad inference block
    cam_np      = gradcam(tensor)
    cam_overlay = gradcam.overlay(cam_np, image.resize((224, 224)))

    with torch.no_grad():
        emb = img_ext(tensor)
    prob = _predict_proba(img_emb=emb)

    label   = "FAKE" if prob >= 0.35 else "REAL"
    dct     = dct_peak_score(tensor.squeeze(0))
    verdict = f"**{label}** — {prob:.1%} fake probability"
    details = f"DCT peak score: {dct:.4f}"
    return verdict, round(prob, 4), details, annotated, cam_overlay


def predict_audio(audio):
    if audio is None:
        return "Upload an audio file to begin.", 0.0
    sr, waveform = audio
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if waveform.max() > 1.0:           # int16 PCM → float
        waveform /= 32768.0
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    target = 4 * 16000
    waveform = np.pad(waveform, (0, max(0, target - len(waveform))))[:target]
    tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = aud_ext(tensor)
    prob = _predict_proba(aud_emb=emb)
    label = "FAKE" if prob >= 0.35 else "REAL"
    return f"**{label}** — {prob:.1%} fake probability", round(prob, 4)


def predict_video(video_path: str):
    if video_path is None:
        return "Upload a video to begin.", 0.0, ""

    cap = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    step  = max(total // CLIP_FRAMES, 1)

    frames_112, frames_224 = [], []
    for i in range(0, total, step):
        if len(frames_112) >= CLIP_FRAMES:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames_112.append(_VID_TRANSFORM(pil))
        frames_224.append(_IMG_TRANSFORM(pil))
    cap.release()

    img_emb = vid_emb = aud_emb = None
    modalities = []

    if frames_112:
        while len(frames_112) < CLIP_FRAMES:   # pad to fixed length
            frames_112.append(frames_112[-1])
        clip  = torch.stack(frames_112).unsqueeze(0).to(device)  # (1, T, C, 112, 112)
        batch = torch.stack(frames_224).to(device)               # (T, C, 224, 224)
        with torch.no_grad():
            vid_emb = vid_ext(clip)
            img_emb = img_ext(batch).mean(0, keepdim=True)
        modalities += ["video", "image"]
        tc = vid_ext.temporal_consistency_score(clip.squeeze(0))
    else:
        tc = None

    # Audio via ffmpeg
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wav_path = wf.name
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
        capture_output=True,
    )
    if res.returncode == 0:
        try:
            wav, _ = sf.read(wav_path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            target = 4 * 16000
            wav = np.pad(wav, (0, max(0, target - len(wav))))[:target]
            t = torch.from_numpy(wav).unsqueeze(0).to(device)
            with torch.no_grad():
                aud_emb = aud_ext(t)
            modalities.append("audio")
        except Exception:
            pass
    Path(wav_path).unlink(missing_ok=True)

    if img_emb is None and vid_emb is None:
        return "Could not extract any features from this video.", 0.0, ""

    prob = _predict_proba(img_emb=img_emb, aud_emb=aud_emb, vid_emb=vid_emb)

    label   = "FAKE" if prob >= 0.35 else "REAL"
    verdict = f"**{label}** — {prob:.1%} fake probability"
    details_parts = [f"Modalities: {', '.join(modalities)}"]
    if tc is not None:
        details_parts.append(f"Temporal consistency: {tc:.4f}")
    return verdict, round(prob, 4), " | ".join(details_parts)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown(
        "# Deepfake Detector\n"
        "Multimodal detection — XceptionNet · LCNN · R3D-18 · Cross-Attention Fusion"
    )

    with gr.Tab("Image"):
        with gr.Row():
            img_in        = gr.Image(type="pil", label="Input image")
            img_annotated = gr.Image(type="pil", label="Face detection", interactive=False)
            img_gradcam   = gr.Image(type="pil", label="Grad-CAM",       interactive=False)
        with gr.Row():
            with gr.Column():
                img_verdict = gr.Markdown("Upload an image to begin.")
                img_score   = gr.Slider(0, 1, value=0, label="Fake probability", interactive=False)
                img_detail  = gr.Textbox(label="Diagnostics", interactive=False)
        gr.Button("Analyse").click(
            predict_image, inputs=img_in,
            outputs=[img_verdict, img_score, img_detail, img_annotated, img_gradcam],
        )

    with gr.Tab("Audio"):
        with gr.Row():
            aud_in = gr.Audio(label="Input audio")
            with gr.Column():
                aud_verdict = gr.Markdown("Upload audio to begin.")
                aud_score   = gr.Slider(0, 1, value=0, label="Fake probability", interactive=False)
        gr.Button("Analyse").click(
            predict_audio, inputs=aud_in,
            outputs=[aud_verdict, aud_score],
        )

    with gr.Tab("Video"):
        with gr.Row():
            vid_in = gr.Video(label="Input video")
            with gr.Column():
                vid_verdict = gr.Markdown("Upload a video to begin.")
                vid_score   = gr.Slider(0, 1, value=0, label="Fake probability", interactive=False)
                vid_detail  = gr.Textbox(label="Diagnostics", interactive=False)
        gr.Button("Analyse").click(
            predict_video, inputs=vid_in,
            outputs=[vid_verdict, vid_score, vid_detail],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if Path(args.checkpoint).exists():
        _load_checkpoint(args.checkpoint)
    else:
        print(f"No checkpoint at {args.checkpoint} — running with untrained weights")

    _build_gradcam()  # must come after checkpoint load to target the right scorer
    demo.launch(share=args.share, server_port=args.port, theme=gr.themes.Soft())
