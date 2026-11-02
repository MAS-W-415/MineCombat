
import json

import numpy as np
import torch

from minecombat.policies.harness.gate import FixedIntervalGate
from minecombat.policies.harness.high_level_ppo import (
    HighLevelPPOConfig,
)
from minecombat.policies.harness.high_level_session import (
    HighLevelRolloutSession,
)
from minecombat.policies.harness.high_level_trainer import (
    HighLevelTrainConfig,
    train_high_level,
)
from minecombat.policies.harness.runtime import HarnessRuntime
from minecombat.policies.harness.tactical_policy import (
    HighLevelTacticalPolicy,
)
from minecombat.policies.harness.visual_features import (
    VisualGridFeatureAdapter,
)


class FakeLowPolicy:
    def get_action(
        self, obs, memory, *,
        tactical_condition, deterministic, input_shape
    ):
        return tactical_condition.tactic_id, memory


class FakeEnv:
    def reset(self):
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        return {
            "image": np.full(
                (16, 16, 3),
                20 + self.steps,
                dtype=np.uint8,
            )
        }

    def step(self, action):
        self.steps += 1
        return self._obs(), 1.0, False, False, {}


def test_high_level_training_loop(tmp_path):
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

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-3,
    )

    before = policy.actor_head.weight.detach().clone()

    history = train_high_level(
        FakeEnv(),
        session,
        optimizer,
        train_config=HighLevelTrainConfig(
            num_updates=2,
            episodes_per_update=2,
            max_steps=5,
        ),
        ppo_config=HighLevelPPOConfig(
            update_epochs=2,
        ),
        save_dir=tmp_path,
    )

    # Two PPO updates.
    assert len(history) == 2

    # 2 updates × 2 episodes × 5 steps.
    assert history[-1]["total_env_steps"] == 20
    assert history[-1]["total_episodes"] == 4

    # Each episode produces 3 high-level transitions.
    assert all(
        record["num_transitions"] == 6
        for record in history
    )

    # The high-level Actor was updated.
    assert not torch.equal(
        before,
        policy.actor_head.weight.detach(),
    )

    checkpoint = tmp_path / "high_level_latest.pt"
    assert checkpoint.exists()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    assert state["update"] == 2
    assert state["total_env_steps"] == 20
    assert "policy" in state
    assert "optimizer" in state

    saved_history = json.loads(
        (tmp_path / "history.json").read_text()
    )

    assert len(saved_history) == 2

    # Old on-policy data must be discarded.
    assert session.collector.transitions == []
    assert session.memory is None
