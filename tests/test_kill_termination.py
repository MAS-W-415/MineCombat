
"""Tests for combat success termination."""

from __future__ import annotations

import pytest

from minecombat.envs.callbacks.kill_termination import (
    KillTerminationCallback,
)


def make_info(kills: float) -> dict:
    return {
        "kill_entity": {
            "zombie": kills,
        },
    }


def run_step(
    callback: KillTerminationCallback,
    kills: float,
    terminated: bool = False,
):
    return callback.after_step(
        sim=None,
        obs={},
        reward=10.0,
        terminated=terminated,
        truncated=False,
        info=make_info(kills),
    )


def test_killing_zombie_terminates_episode():
    callback = KillTerminationCallback(required_kills=1)

    callback.after_reset(None, {}, make_info(0))

    obs, reward, terminated, truncated, info = run_step(
        callback,
        kills=1,
    )

    assert terminated is True
    assert truncated is False
    assert reward == 10.0


def test_no_kill_does_not_terminate():
    callback = KillTerminationCallback(required_kills=1)

    callback.after_reset(None, {}, make_info(0))

    _, _, terminated, truncated, _ = run_step(
        callback,
        kills=0,
    )

    assert terminated is False
    assert truncated is False


def test_previous_episode_kills_are_not_counted():
    callback = KillTerminationCallback(required_kills=1)

    # The cumulative counter already contains five kills.
    callback.after_reset(None, {}, make_info(5))

    # No new kill in this episode.
    _, _, terminated, _, _ = run_step(
        callback,
        kills=5,
    )

    assert terminated is False

    # One additional kill completes the task.
    _, _, terminated, _, _ = run_step(
        callback,
        kills=6,
    )

    assert terminated is True


def test_multiple_required_kills():
    callback = KillTerminationCallback(required_kills=2)

    callback.after_reset(None, {}, make_info(0))

    _, _, first_terminated, _, _ = run_step(
        callback,
        kills=1,
    )

    _, _, second_terminated, _, _ = run_step(
        callback,
        kills=2,
    )

    assert first_terminated is False
    assert second_terminated is True


def test_existing_termination_is_preserved():
    callback = KillTerminationCallback(required_kills=1)

    callback.after_reset(None, {}, make_info(0))

    _, _, terminated, _, _ = run_step(
        callback,
        kills=0,
        terminated=True,
    )

    assert terminated is True


def test_invalid_required_kills():
    with pytest.raises(ValueError):
        KillTerminationCallback(required_kills=0)
