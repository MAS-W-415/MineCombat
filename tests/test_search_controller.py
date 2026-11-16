
import pytest

from minecombat.policies.harness.search_controller import (
    ClosedLoopSearchController,
    SearchConfig,
    TargetDetection,
)


def test_search_align_and_combat():
    controller = ClosedLoopSearchController()

    # The target is initially invisible.
    command, memory = controller.step(TargetDetection())
    assert command.mode == "search"
    assert command.yaw_delta == 10.0

    # Target appears on the right side.
    command, memory = controller.step(
        TargetDetection(center_x=0.5, confidence=0.9),
        memory,
    )
    assert command.mode == "align"
    assert command.yaw_delta == pytest.approx(6.0)

    # Target reaches the center.
    command, memory = controller.step(
        TargetDetection(center_x=0.05, confidence=0.9),
        memory,
    )
    assert command.mode == "combat"
    assert command.use_combat_policy
    assert command.yaw_delta is None


def test_target_loss_and_reacquisition():
    controller = ClosedLoopSearchController()

    # Target was last seen on the left.
    _, memory = controller.step(
        TargetDetection(center_x=-0.5, confidence=1.0)
    )

    # Brief visual loss is tolerated.
    for _ in range(2):
        command, memory = controller.step(TargetDetection(), memory)
        assert command.mode == "combat"

    # After the grace period, search toward the last position.
    command, memory = controller.step(TargetDetection(), memory)
    assert command.mode == "search"
    assert command.yaw_delta == -10.0

    # Seeing the target again resets the missing-frame count.
    command, memory = controller.step(
        TargetDetection(center_x=0.0, confidence=1.0),
        memory,
    )
    assert command.mode == "combat"
    assert memory.misses == 0


def test_search_reverses_direction():
    controller = ClosedLoopSearchController(
        SearchConfig(reverse_after_steps=2)
    )

    memory = None
    yaws = []

    for _ in range(3):
        command, memory = controller.step(TargetDetection(), memory)
        yaws.append(command.yaw_delta)

    assert yaws == [10.0, 10.0, -10.0]


def test_low_confidence_and_episode_reset():
    controller = ClosedLoopSearchController()

    command, memory = controller.step(
        TargetDetection(center_x=0.8, confidence=0.1)
    )
    assert command.mode == "search"

    # None starts a fresh episode with no previous search memory.
    command, fresh_memory = controller.step(
        TargetDetection(), memory=None
    )
    assert command.yaw_delta == 10.0
    assert not fresh_memory.has_seen_target


def test_invalid_detection():
    with pytest.raises(ValueError):
        TargetDetection(center_x=1.5, confidence=0.9)

    with pytest.raises(ValueError):
        TargetDetection(center_x=0.0, confidence=float("nan"))
