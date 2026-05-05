"""
Smoke tests — shape checks only, no real data required.
    pytest tests/
"""
import itertools
import torch
import pytest

device = torch.device("cpu")


@pytest.fixture(scope="module")
def img_ext():
    from detectors.image.extractor import ImageFeatureExtractor
    return ImageFeatureExtractor(pretrained=False).to(device).eval()


@pytest.fixture(scope="module")
def aud_ext():
    from detectors.audio.extractor import AudioFeatureExtractor
    return AudioFeatureExtractor(sample_rate=16000).to(device).eval()


@pytest.fixture(scope="module")
def vid_ext():
    from detectors.video.extractor import VideoFeatureExtractor, CLIP_FRAMES, CLIP_SIZE
    return VideoFeatureExtractor(pretrained=False).to(device).eval()


@pytest.fixture(scope="module")
def fusion():
    from fusion.cross_attention import MultiModalFusionClassifier
    return MultiModalFusionClassifier().to(device).eval()


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def test_image_extractor_shape(img_ext):
    x = torch.zeros(2, 3, 299, 299)
    with torch.no_grad():
        out = img_ext(x)
    assert out.shape == (2, 512)


def test_audio_extractor_shape(aud_ext):
    x = torch.zeros(2, 64000)
    with torch.no_grad():
        out = aud_ext(x)
    assert out.shape == (2, 512)


def test_video_extractor_shape(vid_ext):
    from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE
    x = torch.zeros(2, CLIP_FRAMES, 3, CLIP_SIZE, CLIP_SIZE)
    with torch.no_grad():
        out = vid_ext(x)
    assert out.shape == (2, 512)


def test_video_temporal_score(vid_ext):
    from detectors.video.extractor import CLIP_FRAMES, CLIP_SIZE
    clip = torch.zeros(CLIP_FRAMES, 3, CLIP_SIZE, CLIP_SIZE)
    score = vid_ext.temporal_consistency_score(clip)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Fusion — all 7 non-empty modality combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("combo", [
    c for r in range(1, 4)
    for c in itertools.combinations(["img_emb", "aud_emb", "vid_emb"], r)
])
def test_fusion_modality_combos(fusion, combo):
    kwargs = {k: torch.zeros(2, 512) for k in combo}
    with torch.no_grad():
        out = fusion(**kwargs)
    assert out.shape == (2, 1)


# ---------------------------------------------------------------------------
# DCT score
# ---------------------------------------------------------------------------

def test_dct_peak_score():
    from detectors.image.extractor import dct_peak_score
    img = torch.rand(3, 299, 299)
    score = dct_peak_score(img)
    assert isinstance(score, float)
    assert score >= 0


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

def test_gradcam_shape(img_ext, fusion):
    from utils.gradcam import DeepfakeGradCAM
    cam = DeepfakeGradCAM(img_ext, fusion)
    x = torch.zeros(1, 3, 299, 299, requires_grad=False)
    heatmap = cam(x)
    cam.remove_hooks()
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0 and heatmap.max() <= 1
