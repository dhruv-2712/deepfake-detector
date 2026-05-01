"""
Grad-CAM visualization for the image extractor.

Target layer: the last stage of EfficientNet-B4's block stack.
No extra dependencies — implemented with raw PyTorch hooks.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


class DeepfakeGradCAM:
    """
    Wraps ImageFeatureExtractor + MultiModalFusionClassifier to produce
    Grad-CAM heatmaps over the EfficientNet-B4 backbone's last stage.

    Usage:
        cam = DeepfakeGradCAM(img_ext, fusion)
        heatmap = cam(tensor)             # (H, W) float in [0,1]
        overlay = cam.overlay(heatmap, pil_image)
        cam.remove_hooks()                # call when done to avoid memory leaks
    """

    def __init__(self, img_ext, fusion):
        self.img_ext = img_ext
        self.fusion  = fusion
        self._acts: torch.Tensor | None  = None
        self._grads: torch.Tensor | None = None

        target = img_ext.backbone.blocks[-1]
        self._fwd = target.register_forward_hook(
            lambda m, inp, out: setattr(self, "_acts", out)
        )
        self._bwd = target.register_full_backward_hook(
            lambda m, gin, gout: setattr(self, "_grads", gout[0])
        )

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        """
        tensor: (1, 3, 224, 224), already on the correct device.
        Returns heatmap (H, W) in [0, 1].
        """
        self.img_ext.zero_grad()
        self.fusion.zero_grad()

        with torch.enable_grad():
            emb   = self.img_ext(tensor)
            score = self.fusion(img_emb=emb)
            score.sum().backward()

        acts  = self._acts   # (1, C, H, W)
        grads = self._grads  # (1, C, H, W) or None

        if grads is not None:
            weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1, keepdim=True)   # (1, 1, H, W)
        else:
            cam = acts.mean(dim=1, keepdim=True)              # activation fallback

        cam = torch.relu(cam).squeeze()                       # (H, W)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Zero grads so stray gradients don't affect subsequent training steps
        self.img_ext.zero_grad()
        self.fusion.zero_grad()

        return cam.detach().cpu().numpy()

    def overlay(
        self,
        cam: np.ndarray,
        pil_image: Image.Image,
        size: int = 224,
        alpha: float = 0.45,
    ) -> Image.Image:
        """Blend a JET heatmap over the image. Returns a PIL Image."""
        cam_uint8   = (cam * 255).astype(np.uint8)
        cam_resized = cv2.resize(cam_uint8, (size, size))
        heatmap_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        img_np  = np.array(pil_image.resize((size, size))).astype(np.float32)
        blended = ((1 - alpha) * img_np + alpha * heatmap_rgb).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()
