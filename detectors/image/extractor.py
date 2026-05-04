import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from scipy.fft import dctn


def _build_srm_kernels() -> torch.Tensor:
    """30 SRM high-pass residual filters, shape (30, 1, 5, 5)."""
    # Canonical SRM filter set used in forensics literature.
    # We store one representative set: 3 base kernels tiled to 30 by rotating.
    f1 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  2, -4,  2,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32) / 4.0

    f2 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1],
    ], dtype=np.float32) / 12.0

    f3 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32) / 2.0

    bases = [f1, f2, f3]
    kernels = []
    for b in bases:
        for k in range(10):
            kernels.append(np.roll(b, k, axis=0))
    kernels = np.stack(kernels, axis=0)[:, np.newaxis]  # (30, 1, 5, 5)
    return torch.from_numpy(kernels)


class SRMConv(nn.Module):
    """Single-channel SRM with 30 frozen high-pass filters."""

    def __init__(self):
        super().__init__()
        weight = _build_srm_kernels()  # (30, 1, 5, 5)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — convert to grayscale for SRM
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return F.conv2d(gray, self.weight, padding=2)  # (B, 30, H, W)


def dct_peak_score(image_tensor: torch.Tensor) -> float:
    """2-D DCT on luma channel → log-magnitude → peak std score (scalar)."""
    # image_tensor: (C, H, W) float32 in [0, 1]
    luma = (0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2])
    luma_np = luma.detach().cpu().numpy()
    dct_coeffs = dctn(luma_np, norm="ortho")
    log_mag = np.log1p(np.abs(dct_coeffs))
    # Peak score: std of top-1% coefficients
    flat = log_mag.flatten()
    top_k = flat[flat >= np.percentile(flat, 99)]
    return float(np.std(top_k))


class ImageFeatureExtractor(nn.Module):
    """
    Extracts a (B, 512) embedding from RGB images using:
      - XceptionNet backbone (2048-d global pool features)
      - SRM noise residual path (frozen, adds high-freq artifact signal)
    Input: (B, 3, 299, 299), normalized to [-1, 1].
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "legacy_xception", pretrained=pretrained, num_classes=0
        )  # output: (B, 2048)
        self.srm = SRMConv()
        self.srm_pool = nn.AdaptiveAvgPool2d(1)
        self.srm_proj = nn.Linear(30, 2048)
        self.proj = nn.Linear(2048, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 299, 299)
        cnn_feat = self.backbone(x)                     # (B, 2048)
        srm_maps = self.srm(x)                          # (B, 30, H, W)
        srm_feat = self.srm_pool(srm_maps).flatten(1)   # (B, 30)
        srm_feat = self.srm_proj(srm_feat)              # (B, 2048)
        fused = cnn_feat + srm_feat                     # (B, 2048)
        return self.proj(fused)                         # (B, 512)


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageFeatureExtractor(pretrained=False).to(device)
    model.eval()

    dummy = torch.rand(2, 3, 299, 299, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect (2, 512)

    single = torch.rand(3, 299, 299)
    score = dct_peak_score(single)
    print(f"DCT peak score (random image): {score:.4f}")
