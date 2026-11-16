
"""Tests for the combined visual search and VPT policy."""

import numpy as np
import pytest

from minecombat.policies.harness.search_controller import (
    ClosedLoopSearchController,
    TargetDetection,
)
from minecombat.policies.harness.visual_search_policy import (
    CameraActionAdapter,
    VisualSearchPolicy,
)


class FakeDetector:
    def __init__(self, detections):
        self.detections = iter(detections)
        self.calls = 0

    def detect(self, obs):
        self.calls += 1
        return next(self.detections)


class FakeCombatPolicy:
    def __init__(self):
        self.memory_inputs = []
        self.calls = 0

    def get_action(
        self,
        obs,
        memory,
        *,
        deterministic,
        input_shape,
    ):
        self.memory_inputs.append(memory)
        self.calls += 1

        next_memory = 1 if memory is None else memory + 1

        return {
            "buttons": np.array([7]),
            "camera": np.array([60]),
        }, next_memory


class FakeCameraAction:
    def __init__(self):
        self.yaws = []

    def __call__(self, yaw):
        self.yaws.append(yaw)
        return {
            "buttons": np.array([9]),
            "camera": np.array([75]),
        }


def make_obs():
    return {
        "image": np.zeros((128, 128, 3), dtype=np.uint8)
    }


def make_policy(detections):
    detector = FakeDetector(detections)
    combat = FakeCombatPolicy()
    camera = FakeCameraAction()

    policy = VisualSearchPolicy(
        detector=detector,
        controller=ClosedLoopSearchController(),
        combat_policy=combat,
        camera_action=camera,
    )

    return policy, detector, combat, camera


def test_search_align_then_combat():
    policy, detector, combat, camera = make_policy([
        TargetDetection(),
        TargetDetection(center_x=0.5, confidence=0.9),
        TargetDetection(center_x=0.0, confidence=0.9),
    ])

    memory = None
    modes = []

    for _ in range(3):
        action, memory = policy.get_action(
            make_obs(),
            memory,
            deterministic=True,
            input_shape="*",
        )
        assert "buttons" in action
        assert "camera" in action
        modes.append(memory.mode)

    assert modes == ["search", "align", "combat"]
    assert camera.yaws == pytest.approx([10.0, 6.0])
    assert combat.calls == 1
    assert detector.calls == 3
    assert memory.steps == 3


def test_vpt_memory_is_reset_after_search():
    policy, _, combat, _ = make_policy([
        TargetDetection(center_x=0.0, confidence=1.0),
        TargetDetection(),
        TargetDetection(),
        TargetDetection(),
        TargetDetection(center_x=0.0, confidence=1.0),
    ])

    memory = None
    modes = []

    for _ in range(5):
        _, memory = policy.get_action(make_obs(), memory)
        modes.append(memory.mode)

    assert modes == [
        "combat",
        "combat",
        "combat",
        "search",
        "combat",
    ]

    # Grace period preserves the VPT state.
    # After the search step, VPT restarts from None.
    assert combat.memory_inputs == [None, 1, 2, None]


def test_episode_reset():
    policy, _, combat, camera = make_policy([
        TargetDetection(),
        TargetDetection(),
    ])

    _, memory = policy.get_action(make_obs(), None)

    assert memory.mode == "search"
    assert memory.steps == 1

    # Passing None explicitly starts a new episode.
    _, memory = policy.get_action(make_obs(), None)

    assert memory.mode == "search"
    assert memory.steps == 1
    assert camera.yaws == [10.0, 10.0]
    assert combat.calls == 0


def test_camera_adapter_uses_agent_conversion():
    class FakeEnv:
        action_type = "agent"

        def __init__(self):
            self.received = None

        def env_action_to_agent_action(self, action):
            self.received = action["camera"].copy()

            return {
                "buttons": np.array([[10]]),
                "camera": np.array([[80]]),
            }

    env = FakeEnv()
    adapter = CameraActionAdapter(env, yaw_sign=-1)

    action = adapter(6.0)

    assert env.received.shape == (1, 2)
    assert env.received.tolist() == [[0.0, -6.0]]
    assert action["camera"].shape == (1, 1)
    assert "buttons" in action


def test_rejects_wrong_environment():
    class FakeEnv:
        action_type = "env"

        def env_action_to_agent_action(self, action):
            return action

    with pytest.raises(ValueError):
        CameraActionAdapter(FakeEnv())


def test_rejects_invalid_memory():
    policy, _, _, _ = make_policy([])

    with pytest.raises(TypeError):
        policy.get_action(make_obs(), memory="invalid")


def test_requires_single_frame_input():
    policy, _, _, _ = make_policy([])

    with pytest.raises(ValueError):
        policy.get_action(
            make_obs(),
            memory=None,
            input_shape="BT*",
        )
