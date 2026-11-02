
"""Lightweight observation-only visual features for HH2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class VisualGridFeatureAdapter:
    """Pool an RGB observation into a compact feature vector."""

    def __init__(self, grid_size: int = 8):
        if type(grid_size) is not int or grid_size < 1:
            raise ValueError("grid_size must be positive")

        self.grid_size = grid_size
        self.feature_dim = 3 * grid_size * grid_size

    def extract(self, obs, memory):
        if not isinstance(obs, dict) or "image" not in obs:
            raise ValueError("Expected obs with an image field")

        image = torch.as_tensor(obs["image"])

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected RGB image [H, W, 3]")

        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError("Image cannot be empty")

        # [H, W, 3] -> [1, 3, H, W]
        image = (
            image.permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            / 255.0
        )

        pooled = F.adaptive_avg_pool2d(
            image,
            (self.grid_size, self.grid_size),
        )

        features = (
            pooled.reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        # This initial feature adapter is stateless.
        return features, memory
