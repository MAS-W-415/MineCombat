"""Tests for camera-weighted action entropy."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from minecombat.policies.vpt_entropy import (
    CameraWeightedActionHead,
    apply_camera_entropy_scale,
)


class FixedEntropyHead(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.linear = nn.Linear(1, 1)

    def entropy(self, logits, return_dict: bool = False):
        del logits, return_dict
        return torch.tensor(self.value)


def make_head(scale: float = 1.0) -> CameraWeightedActionHead:
    return CameraWeightedActionHead(
        {
            "buttons": FixedEntropyHead(2.0),
            "camera": FixedEntropyHead(3.0),
        },
        camera_scale=scale,
    )


def test_camera_entropy_is_scaled() -> None:
    head = make_head(scale=4.0)

    assert head.entropy({"buttons": None, "camera": None}).item() == (
        2.0 + 4.0 * 3.0
    )


def test_raw_parts_are_unscaled() -> None:
    head = make_head(scale=4.0)

    parts = head.entropy(
        {"buttons": None, "camera": None},
        return_dict=True,
    )

    assert parts["buttons"].item() == 2.0
    assert parts["camera"].item() == 3.0


def test_state_dict_keys_are_unchanged() -> None:
    head = make_head(scale=4.0)

    assert set(head.state_dict()) == {
        "buttons.linear.weight",
        "buttons.linear.bias",
        "camera.linear.weight",
        "camera.linear.bias",
    }


def test_scale_one_leaves_policy_untouched() -> None:
    policy = SimpleNamespace(pi_head=make_head(scale=1.0))

    result = apply_camera_entropy_scale(policy, 1.0)

    assert result is policy
    assert policy.pi_head.camera_scale == 1.0
