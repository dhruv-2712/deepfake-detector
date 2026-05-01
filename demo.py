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
face_det = FaceDetector(device=device)
gradcam  = DeepfakeGradCAM(img_ext, fusion)

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_VID_TRANSFORM = transforms.Compose([
    transforms.Resize((CLIP_SIZE, CLIP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_checkpoint(path: str):
    state = torch.load(path, map_location=device)
    img_ext.load_state_dict(state.get("img_extractor", {}), strict=False)
    aud_ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
    vid_ext.load_state_dict(state.get("vid_extractor", {}), strict=False)
    fusion.load_state_dict(state.get("fusion", {}), strict=False)
    print(f"Loaded checkpoint: {path}")


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_image(image: Image.Image):
    if image is None:
        return "Upload an image to begin.", 0.0, "", None, None

    image     = image.convert("RGB")
    annotated = face_det.draw_boxes(image)
    crop, face_found = face_det.crop_or_resize(image, size=224)

    tensor = _IMG_TRANSFORM(crop).unsqueeze(0).to(device)

    # Grad-CAM needs gradients — run before the no_grad inference block
    cam_np      = gradcam(tensor)
    cam_overlay = gradcam.overlay(cam_np, crop)

    with torch.no_grad():
        score = fusion(img_emb=img_ext(tensor)).squeeze().item()

    label   = "FAKE" if score >= 0.5 else "REAL"
    dct     = dct_peak_score(tensor.squeeze(0))
    source  = "face crop" if face_found else "full frame (no face detected)"
    verdict = f"**{label}** — {score:.1%} fake probability"
    details = f"Analysed: {source} | DCT peak score: {dct:.4f}"
    return verdict, round(score, 4), details, annotated, cam_overlay


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
        score = fusion(aud_emb=aud_ext(tensor)).squeeze().item()
    label = "FAKE" if score >= 0.5 else "REAL"
    return f"**{label}** — {score:.1%} fake probability", round(score, 4)


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

    with torch.no_grad():
        score = fusion(img_emb=img_emb, aud_emb=aud_emb, vid_emb=vid_emb).squeeze().item()

    label   = "FAKE" if score >= 0.5 else "REAL"
    verdict = f"**{label}** — {score:.1%} fake probability"
    details_parts = [f"Modalities: {', '.join(modalities)}"]
    if tc is not None:
        details_parts.append(f"Temporal consistency: {tc:.4f}")
    return verdict, round(score, 4), " | ".join(details_parts)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown(
        "# Deepfake Detector\n"
        "Multimodal detection — EfficientNet-B4 · LCNN · R3D-18 · Cross-Attention Fusion"
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

    demo.launch(share=args.share, server_port=args.port, theme=gr.themes.Soft())
