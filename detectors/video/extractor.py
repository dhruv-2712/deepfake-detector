import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

CLIP_FRAMES = 16
CLIP_SIZE = 112  # spatial size R3D-18 is pretrained on


class VideoFeatureExtractor(nn.Module):
    """
    Extracts a (B, 512) temporal embedding from a video clip using R3D-18
    pretrained on Kinetics-400.

    Input:  (B, T, C, H, W) — frames already normalized to ImageNet stats
    Output: (B, 512)

    R3D-18 expects (B, C, T, H, W) internally; the permute is handled here.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        r3d = r3d_18(weights=weights)
        self.stem   = r3d.stem
        self.layer1 = r3d.layer1
        self.layer2 = r3d.layer2
        self.layer3 = r3d.layer3
        self.layer4 = r3d.layer4
        self.avgpool = r3d.avgpool  # AdaptiveAvgPool3d((1,1,1))
        self.proj = nn.Linear(512, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)  # (B, 512)
        return self.proj(x)             # (B, 512)

    @torch.no_grad()
    def temporal_consistency_score(self, x: torch.Tensor) -> float:
        """
        Variance of layer2 feature maps across the temporal dim — a cheap
        proxy for how consistently the network activates over time.
        Low variance = smooth, consistent clip (expected for real video).
        """
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        x = self.layer1(x)
        feat = self.layer2(x)          # (1, 128, T', H', W')
        return float(feat.var(dim=2).mean().item())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoFeatureExtractor(pretrained=False).to(device)
    model.eval()

    dummy = torch.rand(2, CLIP_FRAMES, 3, CLIP_SIZE, CLIP_SIZE, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect torch.Size([2, 512])

    clip = torch.rand(CLIP_FRAMES, 3, CLIP_SIZE, CLIP_SIZE, device=device)
    score = model.temporal_consistency_score(clip)
    print(f"Temporal consistency score (random clip): {score:.4f}")
