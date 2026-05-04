import io
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.audio.extractor import AudioFeatureExtractor
from detectors.image.extractor import ImageFeatureExtractor, dct_peak_score
from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE, VideoFeatureExtractor
from fusion.cross_attention import MultiModalFusionClassifier
from utils.gradcam import DeepfakeGradCAM

app = FastAPI(title="Deepfake Detector", version="2.0.0")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_img_ext:  Optional[ImageFeatureExtractor]      = None
_aud_ext:  Optional[AudioFeatureExtractor]      = None
_vid_ext:  Optional[VideoFeatureExtractor]      = None
_head:     Optional[nn.Module]                  = None
_fusion:   Optional[MultiModalFusionClassifier] = None
_device:   torch.device = torch.device("cpu")

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


# ---------------------------------------------------------------------------
# GradCAM adapter: routes a single image embedding through the fusion head
# ---------------------------------------------------------------------------

class _FusionScorer(nn.Module):
    def __init__(self, fusion: MultiModalFusionClassifier):
        super().__init__()
        self.fusion = fusion

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fusion(img_emb=emb)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def load_models():
    global _img_ext, _aud_ext, _vid_ext, _head, _fusion, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _img_ext = ImageFeatureExtractor(pretrained=True).to(_device).eval()
    _aud_ext = AudioFeatureExtractor(sample_rate=16000).to(_device).eval()
    _vid_ext = VideoFeatureExtractor(pretrained=True).to(_device).eval()
    _head = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    ).to(_device).eval()

    # Prefer fusion checkpoint, fall back to image-only
    for ckpt_name in ("checkpoints/best_fusion.pt", "checkpoints/best_audio.pt",
                      "checkpoints/best.pt"):
        ckpt = Path(ckpt_name)
        if not ckpt.exists():
            continue
        state = torch.load(ckpt, map_location=_device)
        _img_ext.load_state_dict(state.get("img_extractor", {}), strict=False)
        _aud_ext.load_state_dict(state.get("aud_extractor", {}), strict=False)
        _vid_ext.load_state_dict(state.get("vid_extractor", {}), strict=False)
        head_state = state.get("head", {})
        if head_state:
            _head.load_state_dict(head_state)
        fusion_state = state.get("fusion", {})
        if fusion_state:
            _fusion = MultiModalFusionClassifier().to(_device).eval()
            _fusion.load_state_dict(fusion_state)
        val_info = f"  val_acc={state['val_acc']:.4f}" if "val_acc" in state else ""
        print(f"Loaded {ckpt_name} (epoch {state.get('epoch', '?')}){val_info}")
        break


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(img_emb=None, aud_emb=None, vid_emb=None) -> dict:
    with torch.no_grad():
        if _fusion is not None:
            logits = _fusion(img_emb=img_emb, aud_emb=aud_emb, vid_emb=vid_emb)
            prob = torch.sigmoid(logits).squeeze().item()
        else:
            emb = img_emb if img_emb is not None else (vid_emb if vid_emb is not None else aud_emb)
            prob = torch.sigmoid(_head(emb)).squeeze().item()
    return {"is_fake": prob >= 0.5, "confidence": round(prob, 4)}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _image_embedding(data: bytes):
    """Returns (emb (1,512), dct_score float, pil Image)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    raw = transforms.ToTensor()(img)          # (3, H, W) in [0,1]
    dct = round(dct_peak_score(raw), 4)
    t = _IMG_TRANSFORM(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        emb = _img_ext(t)
    return emb, dct, img


def _audio_embedding(data: bytes) -> torch.Tensor:
    waveform, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    target = 4 * 16000
    waveform = np.pad(waveform, (0, max(0, target - len(waveform))))[:target]
    t = torch.from_numpy(waveform).unsqueeze(0).to(_device)
    with torch.no_grad():
        return _aud_ext(t)


def _video_embeddings(data: bytes):
    """Returns (img_emb, aud_emb, vid_emb, temporal_score)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        mp4 = f.name

    cap   = cv2.VideoCapture(mp4)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    step  = max(total // CLIP_FRAMES, 1)
    f112, f224 = [], []
    for i in range(0, total, step):
        if len(f112) >= CLIP_FRAMES:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f112.append(_VID_TRANSFORM(pil))
        f224.append(_IMG_TRANSFORM(pil))
    cap.release()

    img_emb = vid_emb = temporal_score = None
    if f112:
        while len(f112) < CLIP_FRAMES:
            f112.append(f112[-1])
        clip_stack = torch.stack(f112)                        # (T, C, H, W)
        clip  = clip_stack.unsqueeze(0).to(_device)           # (1, T, C, H, W)
        batch = torch.stack(f224).to(_device)                 # (T, C, H, W)
        with torch.no_grad():
            vid_emb = _vid_ext(clip)
            img_emb = _img_ext(batch).mean(0, keepdim=True)
        temporal_score = round(
            _vid_ext.temporal_consistency_score(clip_stack.to(_device)), 4
        )

    aud_emb = None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wav = wf.name
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", mp4, "-ar", "16000", "-ac", "1", "-vn", wav],
        capture_output=True,
    )
    if res.returncode == 0:
        try:
            with open(wav, "rb") as fh:
                aud_emb = _audio_embedding(fh.read())
        except Exception:
            pass

    Path(mp4).unlink(missing_ok=True)
    Path(wav).unlink(missing_ok=True)
    return img_emb, aud_emb, vid_emb, temporal_score


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(_device)}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if _head is None and _fusion is None:
        raise HTTPException(503, "Models not loaded")
    try:
        img_emb, dct, _ = _image_embedding(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    result = _score(img_emb=img_emb)
    result["modalities_used"] = ["image"]
    result["dct_score"] = dct
    return JSONResponse(result)


@app.post("/detect/image/heatmap")
async def detect_image_heatmap(file: UploadFile = File(...)):
    """Returns a PNG heatmap overlay showing which regions drove the prediction."""
    if _img_ext is None:
        raise HTTPException(503, "Models not loaded")
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        t = _IMG_TRANSFORM(img).unsqueeze(0).to(_device)
        scorer = _FusionScorer(_fusion) if _fusion is not None else _head
        cam = DeepfakeGradCAM(_img_ext, scorer)
        heatmap = cam(t)
        overlay = cam.overlay(heatmap, img)
        cam.remove_hooks()
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(400, f"Could not generate heatmap: {e}")


@app.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    if _head is None and _fusion is None:
        raise HTTPException(503, "Models not loaded")
    try:
        aud_emb = _audio_embedding(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")
    result = _score(aud_emb=aud_emb)
    result["modalities_used"] = ["audio"]
    return JSONResponse(result)


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if _head is None and _fusion is None:
        raise HTTPException(503, "Models not loaded")
    try:
        img_emb, aud_emb, vid_emb, temporal_score = _video_embeddings(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Could not process video: {e}")
    if img_emb is None and vid_emb is None:
        raise HTTPException(422, "Could not extract any features from video")
    modalities = (
        (["image"] if img_emb is not None else [])
        + (["audio"] if aud_emb is not None else [])
        + (["video"] if vid_emb is not None else [])
    )
    result = _score(img_emb=img_emb, aud_emb=aud_emb, vid_emb=vid_emb)
    result["modalities_used"] = modalities
    if temporal_score is not None:
        result["temporal_consistency"] = temporal_score
    return JSONResponse(result)
