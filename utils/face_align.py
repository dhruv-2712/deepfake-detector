"""
Face detection and cropping utilities.

Primary:  MTCNN from facenet-pytorch (accurate, 5-point landmark alignment).
Fallback: OpenCV Haar cascade (built-in, no extra deps, less accurate).

Usage:
    detector = FaceDetector(device)
    crop, found = detector.crop_or_resize(pil_image)   # always returns something
    faces       = detector.detect_all(pil_image)       # list, may be empty
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from facenet_pytorch import MTCNN as _MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False


class FaceDetector:
    MIN_CONF = 0.90

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        min_face_size: int = 40,
        margin: float = 0.15,
    ):
        self.margin = margin
        if _MTCNN_AVAILABLE:
            self._mtcnn = _MTCNN(
                keep_all=True,
                min_face_size=min_face_size,
                device=device,
                post_process=False,
            )
            self._cascade = None
        else:
            self._mtcnn = None
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_boxes(self, image: Image.Image) -> List[Tuple[list, float]]:
        if self._mtcnn is not None:
            boxes, probs = self._mtcnn.detect(image)
            if boxes is None:
                return []
            return [(b.tolist(), float(p)) for b, p in zip(boxes, probs)
                    if p >= self.MIN_CONF]
        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        return [([x, y, x + w, y + h], 1.0) for x, y, w, h in faces]

    def _crop_box(self, image: Image.Image, box: list, size: int) -> Optional[Image.Image]:
        x1, y1, x2, y2 = [int(v) for v in box]
        W, H = image.size
        pad = int((x2 - x1) * self.margin)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return image.convert("RGB").crop((x1, y1, x2, y2)).resize(
            (size, size), Image.BILINEAR
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(self, image: Image.Image, size: int = 299) -> List[Image.Image]:
        """Returns every detected face as a cropped PIL image."""
        return [
            crop for box, _ in self._raw_boxes(image)
            if (crop := self._crop_box(image, box, size)) is not None
        ]

    def detect_largest(self, image: Image.Image, size: int = 299) -> Optional[Image.Image]:
        """Returns the largest detected face, or None."""
        boxes = self._raw_boxes(image)
        if not boxes:
            return None
        box = max(boxes, key=lambda bp: (bp[0][2] - bp[0][0]) * (bp[0][3] - bp[0][1]))[0]
        return self._crop_box(image, box, size)

    def crop_or_resize(
        self, image: Image.Image, size: int = 299
    ) -> Tuple[Image.Image, bool]:
        """Always returns (image, face_found). Falls back to full-frame resize."""
        face = self.detect_largest(image, size)
        if face is not None:
            return face, True
        return image.convert("RGB").resize((size, size), Image.BILINEAR), False

    def draw_boxes(self, image: Image.Image) -> Image.Image:
        """Returns a copy of the image with bounding boxes drawn (for visualisation)."""
        from PIL import ImageDraw
        out = image.convert("RGB").copy()
        draw = ImageDraw.Draw(out)
        for box, conf in self._raw_boxes(image):
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 14)), f"{conf:.2f}", fill="red")
        return out


if __name__ == "__main__":
    print(f"MTCNN available: {_MTCNN_AVAILABLE}")
    detector = FaceDetector(device=torch.device("cpu"))

    # Smoke test with a blank image — should return no faces, graceful fallback
    blank = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    crop, found = detector.crop_or_resize(blank, size=224)
    print(f"Blank image  — face found: {found}, output size: {crop.size}")

    # Smoke test with a real face image from local file if available
    import sys
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1])
        crop, found = detector.crop_or_resize(img, size=224)
        print(f"Input image  — face found: {found}, crop size: {crop.size}")
        annotated = detector.draw_boxes(img)
        annotated.save("face_detection_out.jpg")
        print("Annotated image saved to face_detection_out.jpg")
