
"""Tests for closed-loop search trace recording."""
import json
import pytest
import numpy as np

from minecombat.policies.harness.search_controller import (
    ClosedLoopSearchController,
    TargetDetection,
)
from minecombat.policies.harness.search_trace import (
    SearchTracePolicy,
)
from minecombat.policies.harness.visual_search_policy import (
    VisualSearchPolicy,
)


class FakeDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def detect(self, obs):
        return next(self.detections)


class FakeCombatPolicy:
    def get_action(
        self, obs, memory, *,
        deterministic, input_shape
    ):
        return {
            "buttons": np.array([0]),
            "camera": np.array([60]),
        }, 1 if memory is None else memory + 1


def camera_action(yaw):
    return {
        "buttons": np.array([0]),
        "camera": np.array([61]),
    }


def test_trace_records_search_and_reacquisition():
    detector = FakeDetector([
        TargetDetection(),
        TargetDetection(center_x=0.5, confidence=0.9),
        TargetDetection(center_x=0.0, confidence=0.9),
        TargetDetection(),
        TargetDetection(),
        TargetDetection(),
        TargetDetection(center_x=-0.5, confidence=0.9),
        TargetDetection(center_x=0.0, confidence=0.9),
    ])

    base = VisualSearchPolicy(
        detector=detector,
        controller=ClosedLoopSearchController(),
        combat_policy=FakeCombatPolicy(),
        camera_action=camera_action,
    )

    policy = SearchTracePolicy(base)
    obs = {"image": np.zeros((128, 128, 3), dtype=np.uint8)}

    memory = None
    for _ in range(8):
        _, memory = policy.get_action(obs, memory)

    records = policy.records()
    assert len(records) == 1

    trace = records[0]
    assert trace["steps"] == 8
    assert trace["search_steps"] == 2
    assert trace["align_steps"] == 2
    assert trace["combat_steps"] == 4
    assert trace["detected_steps"] == 4
    assert trace["first_detection_step"] == 2
    assert trace["first_combat_step"] == 3
    assert trace["reacquisitions"] == 1


def test_new_episode_starts_fresh_trace():
    detector = FakeDetector([
        TargetDetection(),
        TargetDetection(),
    ])

    base = VisualSearchPolicy(
        detector=detector,
        controller=ClosedLoopSearchController(),
        combat_policy=FakeCombatPolicy(),
        camera_action=camera_action,
    )

    policy = SearchTracePolicy(base)
    obs = {"image": np.zeros((128, 128, 3), dtype=np.uint8)}

    policy.get_action(obs, memory=None)
    policy.get_action(obs, memory=None)

    records = policy.records()
    assert len(records) == 2
    assert records[0]["steps"] == 1
    assert records[1]["steps"] == 1


def test_step_history_records_decisions_and_actions(tmp_path):
    detector = FakeDetector([
        TargetDetection(),
        TargetDetection(center_x=0.5, confidence=0.9),
        TargetDetection(center_x=0.0, confidence=0.9),
    ])

    policy = SearchTracePolicy(
        VisualSearchPolicy(
            detector=detector,
            controller=ClosedLoopSearchController(),
            combat_policy=FakeCombatPolicy(),
            camera_action=camera_action,
        )
    )

    obs = {
        "image": np.zeros((128, 128, 3), dtype=np.uint8)
    }

    memory = None
    for _ in range(3):
        _, memory = policy.get_action(obs, memory)

    output = policy.export_steps(
        tmp_path / "search_steps.jsonl",
        seeds=[3000],
    )

    rows = [
        json.loads(line)
        for line in output.read_text().splitlines()
    ]

    assert len(rows) == 3
    assert [row["mode"] for row in rows] == [
        "search", "align", "combat"
    ]

    assert [row["step"] for row in rows] == [1, 2, 3]
    assert all(row["seed"] == 3000 for row in rows)

    assert rows[0]["detected"] is False
    assert rows[0]["center_x"] is None

    assert rows[1]["center_x"] == pytest.approx(0.5)
    assert rows[1]["confidence"] == pytest.approx(0.9)

    # Search/align use the camera adapter, combat uses VPT.
    assert [row["camera_id"] for row in rows] == [
        61, 61, 60
    ]


def test_step_history_separates_episodes(tmp_path):
    detector = FakeDetector([
        TargetDetection(),
        TargetDetection(),
    ])

    policy = SearchTracePolicy(
        VisualSearchPolicy(
            detector=detector,
            controller=ClosedLoopSearchController(),
            combat_policy=FakeCombatPolicy(),
            camera_action=camera_action,
        )
    )

    obs = {
        "image": np.zeros((128, 128, 3), dtype=np.uint8)
    }

    policy.get_action(obs, memory=None)
    policy.get_action(obs, memory=None)

    output = policy.export_steps(
        tmp_path / "steps.jsonl",
        seeds=[2000, 3000],
    )

    rows = [
        json.loads(line)
        for line in output.read_text().splitlines()
    ]

    assert len(rows) == 2
    assert [row["seed"] for row in rows] == [2000, 3000]
    assert [row["step"] for row in rows] == [1, 1]


def test_export_rejects_seed_count_mismatch(tmp_path):
    policy = SearchTracePolicy(
        VisualSearchPolicy(
            detector=FakeDetector([]),
            controller=ClosedLoopSearchController(),
            combat_policy=FakeCombatPolicy(),
            camera_action=camera_action,
        )
    )

    with pytest.raises(ValueError, match="Seed count"):
        policy.export_steps(
            tmp_path / "steps.jsonl",
            seeds=[3000],
        )
