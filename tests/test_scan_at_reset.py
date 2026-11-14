"""Tests for the reset-time camera scan."""

from __future__ import annotations

import pytest

from minecombat.envs.callbacks.scan_at_reset import (
    ScanAtResetCallback,
)


def test_scan_overrides_camera_for_configured_steps() -> None:
    callback = ScanAtResetCallback(
        steps=2,
        yaw_per_step=10.0,
    )
    action = {"camera": (0.0, 0.0), "attack": 1}

    callback.after_reset(None, {}, {})
    callback.before_step(None, action)
    assert action["camera"] == (10.0, 0.0)
    assert action["attack"] == 1

    callback.before_step(None, action)
    assert action["camera"] == (10.0, 0.0)

    callback.before_step(None, action)
    assert action["camera"] == (10.0, 0.0)
    assert action["attack"] == 1


def test_scan_resets_between_episodes() -> None:
    callback = ScanAtResetCallback(steps=1, yaw_per_step=-10.0)
    action = {"camera": (0.0, 0.0)}

    callback.after_reset(None, {}, {})
    callback.before_step(None, action)
    assert action["camera"] == (-10.0, 0.0)

    action["camera"] = (0.0, 0.0)
    callback.after_reset(None, {}, {})
    callback.before_step(None, action)
    assert action["camera"] == (-10.0, 0.0)


def test_scan_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        ScanAtResetCallback(steps=0)
    with pytest.raises(ValueError):
        ScanAtResetCallback(yaw_per_step=0.0)
