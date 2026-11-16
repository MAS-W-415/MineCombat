
"""Ultralytics YOLO backend for Minecraft Zombie detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .target_detector import DetectionBox


class UltralyticsBoxPredictor:
    """Detect Zombie boxes using fine-tuned YOLO weights."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        imgsz: int = 320,
        confidence: float = 0.25,
        model: Any = None,
    ) -> None:
        if type(imgsz) is not int or imgsz < 32:
            raise ValueError("imgsz must be at least 32")

        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")

        if model is None:
            path = Path(checkpoint).expanduser()

            if not path.is_file():
                raise FileNotFoundError(path)

            # Lazy import: local tests do not require Ultralytics.
            from ultralytics import YOLO

            model = YOLO(str(path))

        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence

    def predict(
        self,
        rgb_image: np.ndarray,
    ) -> list[DetectionBox]:
        """Return Zombie bounding boxes in pixel coordinates."""

        image = np.asarray(rgb_image)

        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or image.dtype != np.uint8
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("Expected RGB uint8 image [H, W, 3]")

        # Ultralytics expects BGR for numpy image inputs.
        bgr = np.ascontiguousarray(image[:, :, ::-1])

        kwargs = {
            "source": bgr,
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)

        if len(results) != 1:
            raise RuntimeError("Expected exactly one image result")

        result = results[0]
        names = result.names

        # Do not silently treat COCO classes as Zombie.
        if "zombie" not in {
            str(name).lower() for name in names.values()
        }:
            raise ValueError(
                "The model has no zombie class. "
                "Load fine-tuned Minecraft detection weights."
            )

        if result.boxes is None:
            return []

        coords = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()

        detections = []

        for xyxy, score, class_id in zip(
            coords, scores, classes, strict=True
        ):
            label = str(
                names[int(class_id)]
            ).lower()

            if label != "zombie":
                continue

            detections.append(
                DetectionBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=float(score),
                    label="zombie",
                )
            )

        return detections
