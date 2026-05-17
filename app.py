"""
Hugging Face Spaces entry point.

Set the HF_MODEL_REPO environment variable (or edit the default below)
to point at the HF Hub repo where best.pt is stored.
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

# ── download weights on cold start ─────────────────────────────────────────
_REPO_ID  = os.getenv("HF_MODEL_REPO", "dhruv2712/deepfake-detector-weights")
_CKPT_DIR = Path("checkpoints")
_CKPT_DIR.mkdir(exist_ok=True)
_CKPT_PATH = _CKPT_DIR / "best.pt"

if not _CKPT_PATH.exists():
    print(f"Downloading weights from {_REPO_ID} …")
    hf_hub_download(
        repo_id=_REPO_ID,
        filename="best.pt",
        local_dir=str(_CKPT_DIR),
    )
    print("Weights downloaded.")

# ── load the demo (reuses all of demo.py) ──────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from demo import _load_checkpoint, _build_gradcam, demo   # noqa: E402

_load_checkpoint(str(_CKPT_PATH))
_build_gradcam()

demo.launch()
