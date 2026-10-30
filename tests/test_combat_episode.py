"""Tests for unified combat episode outcomes."""

from __future__ import annotations

import pytest

from minecombat.envs.callbacks.combat_episode import CombatEpisodeCallback


def make_info(
    health: float = 20.0,
    kills: float = 0.0,
    time_since_death: float | None = None,
) -> dict:
    info = {
        "health": health,
        "kill_entity": {
            "zombie": kills,
        },
    }
    if time_since_death is not None:
        info["custom"] = {"time_since_death": time_since_death}
    return info


def run_step(
    callback: CombatEpisodeCallback,
    health: float = 20.0,
    kills: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    time_since_death: float | None = None,
):
    return callback.after_step(
        sim=None,
        obs={"image": "fake"},
        reward=3.5,
        terminated=terminated,
        truncated=truncated,
        info=make_info(health, kills, time_since_death),
    )


def test_initial_episode_state():
    callback = CombatEpisodeCallback()

    _, info = callback.after_reset(
        None, {}, make_info()
    )

    assert info["combat_outcome"] == "ongoing"
    assert info["combat_success"] is False
    assert info["combat_episode_steps"] == 0


def test_success_terminates_episode():
    callback = CombatEpisodeCallback(required_kills=1)
    callback.after_reset(None, {}, make_info())

    obs, reward, terminated, truncated, info = run_step(
        callback,
        kills=1,
    )

    assert terminated is True
    assert truncated is False
    assert info["combat_outcome"] == "success"
    assert info["combat_success"] is True
    assert info["combat_kills"] == 1.0
    assert reward == 3.5
    assert obs == {"image": "fake"}


def test_player_death_terminates_episode():
    callback = CombatEpisodeCallback()
    callback.after_reset(None, {}, make_info())

    _, _, terminated, truncated, info = run_step(
        callback,
        health=0,
    )

    assert terminated is True
    assert truncated is False
    assert info["combat_outcome"] == "death"
    assert info["combat_success"] is False


def test_death_by_time_since_death_reset():
    callback = CombatEpisodeCallback()
    callback.after_reset(None, {}, make_info(time_since_death=10))

    _, _, terminated, truncated, info = run_step(
        callback,
        health=20,
        time_since_death=0,
    )

    assert terminated is True
    assert truncated is False
    assert info["combat_outcome"] == "death"


def test_increasing_time_since_death_is_not_death():
    callback = CombatEpisodeCallback()
    callback.after_reset(None, {}, make_info(time_since_death=5))

    _, _, terminated, _, info = run_step(
        callback,
        health=20,
        time_since_death=6,
    )

    assert terminated is False
    assert info["combat_outcome"] == "ongoing"


def test_simultaneous_death_and_kill_counts_as_failure():
    callback = CombatEpisodeCallback()
    callback.after_reset(None, {}, make_info())

    _, _, terminated, truncated, info = run_step(
        callback,
        health=0,
        kills=1,
    )

    assert terminated is True
    assert truncated is False
    assert info["combat_outcome"] == "death"
    assert info["combat_success"] is False


def test_timeout_sets_truncated():
    callback = CombatEpisodeCallback(max_episode_steps=3)
    callback.after_reset(None, {}, make_info())

    for _ in range(2):
        _, _, terminated, truncated, info = run_step(
            callback
        )

        assert terminated is False
        assert truncated is False
        assert info["combat_outcome"] == "ongoing"

    _, _, terminated, truncated, info = run_step(callback)

    assert terminated is False
    assert truncated is True
    assert info["combat_outcome"] == "timeout"
    assert info["combat_episode_steps"] == 3


def test_previous_episode_kills_are_ignored():
    callback = CombatEpisodeCallback(required_kills=1)

    callback.after_reset(None, {}, make_info(kills=5))

    _, _, terminated, _, info = run_step(
        callback,
        kills=5,
    )

    assert terminated is False
    assert info["combat_kills"] == 0.0

    _, _, terminated, _, info = run_step(
        callback,
        kills=6,
    )

    assert terminated is True
    assert info["combat_outcome"] == "success"


def test_reset_clears_step_counter():
    callback = CombatEpisodeCallback(max_episode_steps=2)

    callback.after_reset(None, {}, make_info())
    run_step(callback)

    assert callback.episode_steps == 1

    callback.after_reset(None, {}, make_info())

    assert callback.episode_steps == 0


def test_external_termination_is_preserved():
    callback = CombatEpisodeCallback()
    callback.after_reset(None, {}, make_info())

    _, _, terminated, truncated, info = run_step(
        callback,
        terminated=True,
        truncated=True,
    )

    assert terminated is True
    assert truncated is True
    assert info["combat_outcome"] == "external_done"
    assert info["combat_success"] is False


def test_invalid_configuration():
    with pytest.raises(ValueError):
        CombatEpisodeCallback(required_kills=0)

    with pytest.raises(ValueError):
        CombatEpisodeCallback(max_episode_steps=0)