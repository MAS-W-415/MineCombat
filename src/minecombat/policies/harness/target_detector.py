
"""Convert RGB object detections into search-controller signals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .search_controller import TargetDetection


@dataclass(frozen=True)
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "zombie"

    def __post_init__(self):
        values = (
            self.x1, self.y1, self.x2,
            self.y2, self.confidence,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Detection values must be finite")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Invalid bounding box")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Invalid confidence")


class BoxPredictor(Protocol):
    """Backend that predicts boxes directly from RGB pixels."""

    def predict(
        self,
        rgb_image: np.ndarray,
    ) -> Sequence[DetectionBox]:
        ...


class RGBTargetDetector:
    """Select the best visible zombie for the search controller."""

    def __init__(
        self,
        predictor: BoxPredictor,
        *,
        min_confidence: float = 0.5,
        min_area_fraction: float = 0.002,
    ):
        if not 0 <= min_confidence <= 1:
            raise ValueError("Invalid confidence threshold")
        if not 0 <= min_area_fraction < 1:
            raise ValueError("Invalid area threshold")

        self.predictor = predictor
        self.min_confidence = min_confidence
        self.min_area_fraction = min_area_fraction

    def detect(
        self,
        obs: Mapping[str, Any],
    ) -> TargetDetection:
        if "image" not in obs:
            raise ValueError("Observation must contain image")

        image = np.asarray(obs["image"])

        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or image.dtype != np.uint8
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("Expected RGB uint8 image [H, W, 3]")

        height, width = image.shape[:2]
        candidates = []

        for box in self.predictor.predict(image):
            if box.label != "zombie":
                continue

            if box.confidence < self.min_confidence:
                continue

            # Clip predicted boxes to the actual image.
            x1 = max(0.0, min(float(width), box.x1))
            x2 = max(0.0, min(float(width), box.x2))
            y1 = max(0.0, min(float(height), box.y1))
            y2 = max(0.0, min(float(height), box.y2))

            if x2 <= x1 or y2 <= y1:
                continue

            area_fraction = (
                (x2 - x1) * (y2 - y1)
                / (width * height)
            )

            if area_fraction < self.min_area_fraction:
                continue

            center_x = (x1 + x2) / width - 1.0

            candidates.append(
                (box.confidence, area_fraction, center_x)
            )

        if not candidates:
            return TargetDetection()

        # Confidence first, bounding-box area as tie-breaker.
        confidence, _, center_x = max(candidates)

        return TargetDetection(
            center_x=center_x,
            confidence=confidence,
        )
