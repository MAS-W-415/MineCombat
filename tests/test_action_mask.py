"""Tests for stage-1 action masking."""

from __future__ import annotations

from minecombat.envs.callbacks.action_mask import ActionMaskCallback


def test_inventory_is_masked() -> None:
    callback = ActionMaskCallback(mask_inventory=True)
    action = {"inventory": 1, "attack": 1}

    result = callback.before_step(None, action)

    assert result is action
    assert action["inventory"] == 0
    assert action["attack"] == 1


def test_inventory_can_be_left_enabled() -> None:
    callback = ActionMaskCallback(mask_inventory=False)
    action = {"inventory": 1}

    callback.before_step(None, action)

    assert action["inventory"] == 1
