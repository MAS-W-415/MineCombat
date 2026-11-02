
import numpy as np
import torch

from minecombat.policies.harness.runtime import HarnessRuntime
from minecombat.policies.harness.gate import FixedIntervalGate
from minecombat.policies.harness.visual_features import (
    VisualGridFeatureAdapter,
)
from minecombat.policies.harness.tactical_policy import (
    HighLevelTacticalPolicy,
)
from minecombat.policies.harness.high_level_session import (
    HighLevelRolloutSession,
)
from minecombat.policies.harness.episode_collector import (
    collect_episode,
)


class FakeLowPolicy:
    def get_action(
        self, obs, memory, *,
        tactical_condition, deterministic, input_shape
    ):
        return tactical_condition.tactic_id, memory


class FakeEnv:
    def reset(self):
        self.step_count = 0
        return self._obs(), {}

    def _obs(self):
        return {
            "image": np.full(
                (16, 16, 3),
                20 + self.step_count,
                dtype=np.uint8,
            )
        }

    def step(self, action):
        self.step_count += 1

        return self._obs(), 1.0, False, False, {}


def test_episode_collection_truncation():
    torch.manual_seed(0)

    adapter = VisualGridFeatureAdapter(grid_size=8)

    policy = HighLevelTacticalPolicy(
        feature_dim=adapter.feature_dim,
        num_tactics=2,
        hidden_dim=32,
    )

    runtime = HarnessRuntime(
        tactical_policy=policy,
        low_policy=FakeLowPolicy(),
        gate=FixedIntervalGate(interval_steps=2),
        feature_adapter=adapter,
    )

    session = HighLevelRolloutSession(runtime)

    result = collect_episode(
        FakeEnv(),
        session,
        max_steps=5,
    )

    transitions = session.collector.transitions

    assert result.steps == 5
    assert result.total_reward == 5.0
    assert not result.terminated
    assert result.truncated

    # Decisions at steps 1, 3 and 5.
    assert len(transitions) == 3
    assert [t.duration for t in transitions] == [2, 2, 1]

    assert transitions[-1].truncated
    assert not transitions[-1].terminated

    assert all(
        t.behavior_log_prob is not None
        and t.behavior_value is not None
        for t in transitions
    )

    # Next episode starts with clean recurrent memory.
    assert session.memory is None

    session.clear()
    assert session.collector.transitions == []
