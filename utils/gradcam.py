"""
Grad-CAM visualization for the image extractor.

Target layer: the last stage of XceptionNet's block stack.
No extra dependencies — implemented with raw PyTorch hooks.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def _resolve_target_layer(img_ext):
    """
    Return the last spatially-rich layer for Grad-CAM.

    We need a layer whose output is (B, C, H, W) with H, W > 1, so
    that the spatial average of gradients is meaningful.

    Priority:
      1. legacy_xception via timm: backbone.block12 (last middle-flow block)
      2. Any named block* module on the backbone (largest index wins)
      3. EfficientNet: backbone.blocks[-1]
      4. Last Conv2d found by walking the backbone's named modules
    """
    backbone = getattr(img_ext, "backbone", img_ext)

    # 1. timm legacy_xception: block12 is the last middle-flow separable block
    if hasattr(backbone, "block12"):
        return backbone.block12

    # 2. Any backbone with numbered block attributes (block0..blockN)
    block_layers = [
        (name, mod)
        for name, mod in backbone.named_children()
        if name.startswith("block")
    ]
    if block_layers:
        # highest-numbered block
        block_layers.sort(key=lambda t: t[0])
        return block_layers[-1][1]

    # 3. EfficientNet / similar: backbone.blocks sequence
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]

    # 4. Generic fallback: last Conv2d in the backbone
    last_conv = None
    for mod in backbone.modules():
        if isinstance(mod, torch.nn.Conv2d):
            last_conv = mod
    if last_conv is not None:
        return last_conv

    # 5. Ultimate fallback
    children = list(backbone.children())
    return children[-2] if len(children) >= 2 else children[-1]


class DeepfakeGradCAM:
    """
    Wraps ImageFeatureExtractor + head (nn.Module) to produce
    Grad-CAM heatmaps over the XceptionNet backbone's last stage.

    Usage:
        cam = DeepfakeGradCAM(img_ext, head)
        heatmap = cam(tensor)             # (H, W) float in [0,1]
        overlay = cam.overlay(heatmap, pil_image)
        cam.remove_hooks()                # call when done to avoid memory leaks
    """

    def __init__(self, img_ext, head):
        self.img_ext = img_ext
        self.head    = head
        self._acts: torch.Tensor | None  = None
        self._grads: torch.Tensor | None = None

        target = _resolve_target_layer(img_ext)
        self._fwd = target.register_forward_hook(
            lambda m, inp, out: setattr(self, "_acts", out)
        )
        self._bwd = target.register_full_backward_hook(
            lambda m, gin, gout: setattr(self, "_grads", gout[0])
        )

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        """
        tensor: (1, 3, 299, 299), already on the correct device.
        Returns heatmap (H, W) in [0, 1].
        """
        self.img_ext.zero_grad()
        self.head.zero_grad()

        with torch.enable_grad():
            emb   = self.img_ext(tensor)
            score = self.head(emb)
            score.sum().backward()

        acts  = self._acts   # (1, C, H, W)
        grads = self._grads  # (1, C, H, W) or None

        # Guard: we need 4-D spatial tensors for the weighted sum.
        # If the hooked layer squeezed spatial dims (e.g. 1×1 after GAP),
        # fall back to the activation-mean path.
        has_spatial = acts is not None and acts.ndim == 4 and acts.shape[2] > 1

        if has_spatial and grads is not None and grads.ndim == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1, keepdim=True)   # (1, 1, H, W)
        elif has_spatial:
            cam = acts.mean(dim=1, keepdim=True)              # activation fallback
        else:
            # Last resort: return a flat zero heatmap
            cam = torch.zeros(1, 1, 10, 10)

        cam = torch.relu(cam).squeeze()                       # (H, W)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Zero grads so stray gradients don't affect subsequent training steps
        self.img_ext.zero_grad()
        self.head.zero_grad()

        return cam.detach().cpu().numpy()

    def overlay(
        self,
        cam: np.ndarray,
        pil_image: Image.Image,
        size: int = 299,
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
