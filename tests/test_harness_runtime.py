"""Minimal integration test for the hierarchical runtime."""

from minecombat.policies.harness import (
    FixedIntervalGate,
    HarnessRuntime,
    TacticalDecision,
)


class FakeHighPolicy:
    def decide(
        self,
        features,
        risk,
        memory,
        *,
        deterministic,
    ):
        next_id = 0 if memory is None else memory

        return (
            TacticalDecision(tactic_id=next_id),
            next_id + 1,
        )


class FakeLowPolicy:
    def __init__(self):
        self.calls = []

    def get_action(
        self,
        obs,
        memory,
        *,
        tactical_condition,
        deterministic,
        input_shape,
    ):
        self.calls.append(
            (tactical_condition.tactic_id, memory)
        )

        next_memory = 1 if memory is None else memory + 1

        action = {
            "tactic_id": tactical_condition.tactic_id,
        }

        return action, next_memory


def test_hierarchical_scheduling_and_reset():
    low_policy = FakeLowPolicy()

    runtime = HarnessRuntime(
        tactical_policy=FakeHighPolicy(),
        low_policy=low_policy,
        gate=FixedIntervalGate(interval_steps=2),
    )

    memory = None
    tactics = []
    updates = []

    for step in range(5):
        action, memory = runtime.get_action(
            obs={"frame": step},
            memory=memory,
        )

        tactics.append(action["tactic_id"])
        updates.append(memory.last_decision_updated)

    assert tactics == [0, 0, 1, 1, 2]
    assert updates == [True, False, True, False, True]
    assert memory.total_steps == 5

    # A new episode must start without the old episode memory.
    action, new_memory = runtime.get_action(
        obs={"frame": 0},
        memory=None,
    )

    assert action["tactic_id"] == 0
    assert new_memory.total_steps == 1
    assert low_policy.calls[-1] == (0, None)