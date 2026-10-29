"""Tests for combat evaluation rollouts."""

from __future__ import annotations

import pytest

from minecombat.eval.metrics import summarize
from minecombat.eval.rollout import evaluate


def make_info(
    health=20.0,
    kills=0.0,
    damage=0.0,
    outcome="ongoing",
):
    return {
        "health": health,
        "kill_entity": {"zombie": kills},
        "damage_dealt": {"zombie": damage},
        "combat_outcome": outcome,
    }


class FakeCombatEnv:
    """A deterministic simulator with predefined episodes."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.episode_index = -1

    def reset(self):
        self.episode_index += 1
        self.step_index = 0

        episode = self.episodes[self.episode_index]

        return {}, dict(episode["initial"])

    def step(self, action):
        episode = self.episodes[self.episode_index]

        step = episode["steps"][self.step_index]
        self.step_index += 1

        return (
            {},
            step.get("reward", 0.0),
            step.get("terminated", False),
            step.get("truncated", False),
            dict(step["info"]),
        )


def test_success_and_cumulative_damage():
    env = FakeCombatEnv([
        {
            "initial": make_info(),
            "steps": [
                {
                    "info": make_info(
                        health=14,
                        damage=4,
                    ),
                },
                {
                    "info": make_info(
                        health=20,
                        damage=4,
                    ),
                },
                {
                    "info": make_info(
                        health=17,
                        kills=1,
                        damage=8,
                        outcome="success",
                    ),
                    "reward": 10.0,
                    "terminated": True,
                },
            ],
        }
    ])

    records = evaluate(
        env,
        policy=lambda obs: 0,
        episodes=1,
        seed_start=42,
    )

    record = records[0]

    assert record.seed == 42
    assert record.win is True
    assert record.outcome == "success"
    assert record.steps == 3

    # HP: 20 -> 14 -> 20 -> 17
    # Total observed damage: 6 + 0 + 3 = 9.
    assert record.hp_lost == pytest.approx(9.0)

    assert record.damage_dealt == pytest.approx(8.0)
    assert record.time_to_kill == 3
    assert record.reward == pytest.approx(10.0)

    summary = summarize(records)

    assert summary["mean_damage_taken"] == pytest.approx(9.0)


def test_simultaneous_death_and_kill_is_failure():
    env = FakeCombatEnv([
        {
            "initial": make_info(),
            "steps": [
                {
                    "info": make_info(
                        health=0,
                        kills=1,
                        outcome="death",
                    ),
                    "terminated": True,
                },
            ],
        }
    ])

    record = evaluate(
        env,
        policy=lambda obs: 0,
        episodes=1,
    )[0]

    assert record.win is False
    assert record.death is True
    assert record.timeout is False
    assert record.outcome == "death"
    assert record.time_to_kill is None


def test_timeout_external_done_and_eval_limit():
    env = FakeCombatEnv([
        {
            "initial": make_info(),
            "steps": [
                {
                    "info": make_info(outcome="timeout"),
                    "truncated": True,
                },
            ],
        },
        {
            "initial": make_info(),
            "steps": [
                {
                    "info": make_info(
                        outcome="external_done"
                    ),
                    "terminated": True,
                },
            ],
        },
        {
            "initial": make_info(),
            "steps": [
                {"info": make_info()},
                {"info": make_info()},
            ],
        },
    ])

    records = evaluate(
        env,
        policy=lambda obs: 0,
        episodes=3,
        seed_start=10,
        max_steps=2,
    )

    assert [r.seed for r in records] == [10, 11, 12]

    assert records[0].outcome == "timeout"
    assert records[0].timeout is True

    assert records[1].outcome == "external_done"
    assert records[1].timeout is False

    assert records[2].outcome == "eval_limit"
    assert records[2].timeout is True

    summary = summarize(records)

    assert summary["timeout_rate"] == pytest.approx(2 / 3)
    assert summary["external_done_rate"] == pytest.approx(1 / 3)
    assert summary["eval_limit_rate"] == pytest.approx(1 / 3)


def test_invalid_evaluation_arguments():
    env = FakeCombatEnv([])

    with pytest.raises(ValueError):
        evaluate(env, episodes=0)

    with pytest.raises(ValueError):
        evaluate(env, max_steps=0)