
"""Tests for visual search evaluation entry point."""

import json

import numpy as np
import pytest

from minecombat.eval.visual_search_cli import (
    main,
    run_search_evaluation,
)
from minecombat.policies.harness.target_detector import DetectionBox


class FakeEnv:
    action_type = "agent"

    def __init__(self):
        self.step_count = 0

    def env_action_to_agent_action(self, action):
        assert action["camera"].shape == (1, 2)
        return {
            "buttons": np.array([0]),
            "camera": np.array([61]),
        }

    def reset(self):
        self.step_count = 0
        return {
            "image": np.zeros((128, 128, 3), dtype=np.uint8)
        }, {"health": 20.0}

    def step(self, action):
        assert "buttons" in action
        assert "camera" in action

        self.step_count += 1
        done = self.step_count == 3

        return (
            {"image": np.zeros((128, 128, 3), dtype=np.uint8)},
            -0.001,
            False,
            done,
            {
                "health": 20.0,
                "combat_outcome": "timeout" if done else "ongoing",
            },
        )


class FakeCombat:
    def get_action(
        self, obs, memory, *,
        deterministic, input_shape
    ):
        return {
            "buttons": np.array([2]),
            "camera": np.array([60]),
        }, 1


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, rgb_image):
        self.calls += 1

        # First frame of each episode: no detection.
        if self.calls % 3 == 1:
            return []

        return [
            DetectionBox(
                56, 30, 72, 90, 0.9, "zombie"
            )
        ]


def test_evaluation_records_combat_and_search():
    env = FakeEnv()

    records, traces, combat, search = run_search_evaluation(
        env,
        FakeCombat(),
        FakePredictor(),
        episodes=2,
        max_steps=3,
    )

    assert len(records) == 2
    assert len(traces) == 2
    assert combat["episodes"] == 2

    assert search["model_detection_episode_rate"] == 1.0
    assert search["search_step_rate"] == pytest.approx(1 / 3)
    assert search["combat_step_rate"] == pytest.approx(2 / 3)
    assert search["mean_first_detection_step"] == 2

    assert all(t["steps"] == 3 for t in traces)
    assert all(t["first_combat_step"] == 2 for t in traces)


def test_cli_dry_run_without_models(capsys):
    result = main([
        "--combat-checkpoint", "missing_vpt.ckpt",
        "--detector-checkpoint", "missing_yolo.pt",
        "--dry-run",
    ])

    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["episodes"] == 5
    assert output["yaw_sign"] == 1


def test_cli_rejects_invalid_parameters():
    with pytest.raises(SystemExit) as error:
        main([
            "--combat-checkpoint", "missing_vpt.ckpt",
            "--detector-checkpoint", "missing_yolo.pt",
            "--yaw-sign", "0",
            "--dry-run",
        ])

    assert error.value.code == 2
