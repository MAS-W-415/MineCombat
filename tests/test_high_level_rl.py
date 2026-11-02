
"""Regression tests for hierarchical tactical RL."""

import numpy as np
import pytest
import torch

from minecombat.policies.harness import (
    FixedIntervalGate,
    HarnessRuntime,
)
from minecombat.policies.harness.high_level_ppo import (
    HighLevelPPOConfig,
    compute_smdp_gae,
    update_high_level_ppo,
)
from minecombat.policies.harness.high_level_session import (
    HighLevelRolloutSession,
)
from minecombat.policies.harness.tactical_policy import (
    HighLevelTacticalPolicy,
)
from minecombat.policies.harness.tactical_rollout import (
    TacticalTransition,
)
from minecombat.policies.harness.visual_features import (
    VisualGridFeatureAdapter,
)


class FakeLowPolicy:
    """Simulate conditioned execution without Minecraft."""

    def get_action(
        self, obs, memory, *,
        tactical_condition, deterministic, input_shape
    ):
        return {
            "tactic_id": tactical_condition.tactic_id
        }, memory


def make_frame(step):
    return {
        "image": np.full(
            (16, 16, 3),
            20 + step * 10,
            dtype=np.uint8,
        )
    }


def test_harness_high_level_ppo_round_trip():
    torch.manual_seed(0)

    adapter = VisualGridFeatureAdapter(grid_size=8)

    policy = HighLevelTacticalPolicy(
        feature_dim=adapter.feature_dim,
        num_tactics=4,
        hidden_dim=32,
    )

    runtime = HarnessRuntime(
        tactical_policy=policy,
        low_policy=FakeLowPolicy(),
        gate=FixedIntervalGate(interval_steps=2),
        feature_adapter=adapter,
    )

    session = HighLevelRolloutSession(runtime)

    for step in range(5):
        action = session.act(make_frame(step))

        assert 0 <= action["tactic_id"] < 4

        session.observe(
            make_frame(step + 1),
            reward=1.0 + step * 0.1,
            terminated=(step == 4),
        )

    transitions = session.collector.transitions

    assert len(transitions) == 3
    assert [t.duration for t in transitions] == [2, 2, 1]

    assert all(
        t.behavior_log_prob is not None
        and t.behavior_value is not None
        for t in transitions
    )

    before = policy.actor_head.weight.detach().clone()

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-3,
    )

    metrics = update_high_level_ppo(
        policy,
        transitions,
        optimizer,
        HighLevelPPOConfig(update_epochs=2),
    )

    after = policy.actor_head.weight.detach()

    assert metrics["num_transitions"] == 3
    assert not torch.equal(before, after)
    assert session.memory is None

    # Collected data can be cleared after the episode.
    session.clear()
    assert session.collector.transitions == []


def test_truncation_bootstraps_without_crossing_episode():
    """Truncation permits bootstrap but stops the GAE trace."""

    transitions = [
        TacticalTransition(
            features="s0",
            tactic_id=0,
            rewards=(1.0, 2.0),
            next_features="s1",
            terminated=False,
            truncated=True,
        ),
        TacticalTransition(
            features="s2",
            tactic_id=1,
            rewards=(100.0,),
            next_features="s3",
            terminated=True,
            truncated=False,
        ),
    ]

    advantages, returns = compute_smdp_gae(
        transitions,
        old_values=torch.tensor([0.0, 0.0]),
        next_values=torch.tensor([5.0, 0.0]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    # 1 + 0.9*2 + 0.9**2*5 = 6.85
    # The second episode's reward must not leak back.
    assert advantages[0].item() == pytest.approx(6.85)
    assert returns[0].item() == pytest.approx(6.85)

    assert advantages[1].item() == pytest.approx(100.0)
