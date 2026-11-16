
"""Export policy RGB frames for object-detection annotation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _save_png(path: Path, rgb: np.ndarray) -> None:
    """Write RGB pixels as PNG using OpenCV's BGR convention."""
    import cv2

    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Could not save frame: {path}")


class PolicyFrameExporter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        stride: int = 10,
    ) -> None:
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")

        self.root = Path(output_dir)
        self.stride = stride
        self.records: list[dict[str, Any]] = []

    def capture(
        self,
        obs: Mapping[str, Any],
        *,
        seed: int,
        step: int,
    ) -> Path | None:
        """Export reset frame and every Nth environment frame."""
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be non-negative")
        if type(step) is not int or step < 0:
            raise ValueError("step must be non-negative")

        if step % self.stride != 0:
            return None

        if not isinstance(obs, Mapping) or "image" not in obs:
            raise ValueError("Expected observation with image")

        rgb = np.asarray(obs["image"])

        if (
            rgb.ndim != 3
            or rgb.shape[-1] != 3
            or rgb.dtype != np.uint8
            or rgb.shape[0] == 0
            or rgb.shape[1] == 0
        ):
            raise ValueError("Expected uint8 RGB image [H, W, 3]")

        height, width = rgb.shape[:2]

        relative = (
            Path(f"seed_{seed:04d}")
            / f"step_{step:05d}.png"
        )
        path = self.root / relative

        if path.exists():
            raise FileExistsError(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        _save_png(path, rgb)

        self.records.append({
            "image": relative.as_posix(),
            "seed": seed,
            "step": step,
            "width": width,
            "height": height,
        })

        return path

    def write_manifest(self) -> Path:
        """Save frame metadata without creating artificial labels."""
        self.root.mkdir(parents=True, exist_ok=True)

        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            raise FileExistsError(manifest)

        with manifest.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record) + "\n")

        return manifest
