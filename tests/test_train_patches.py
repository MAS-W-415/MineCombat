"""Tests for MineStudio runtime compatibility patches."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
scaled_mse_head = pytest.importorskip(
    "minestudio.utils.vpt_lib.scaled_mse_head"
)

from minecombat.train.patches import patch_scaled_mse_head  # noqa: E402


def test_singleton_value_target_matches_flat_target():
    patch_scaled_mse_head()
    head = scaled_mse_head.ScaledMSEHead(4, 1)
    head.eval()

    prediction = torch.zeros(1, 64)
    flat_target = torch.ones(1, 64)
    extra_dim_target = flat_target.reshape(1, 1, 64)

    flat_loss = head.loss(prediction, flat_target)
    extra_dim_loss = head.loss(prediction, extra_dim_target)

    assert torch.isfinite(extra_dim_loss)
    assert extra_dim_loss.item() == pytest.approx(flat_loss.item())
