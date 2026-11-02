
"""Tests for building a trainable H2 harness."""

import numpy as np
import pytest
import torch
from torch import nn

from minecombat.policies.harness import (
    FixedIntervalGate,
    HarnessRuntime,
    high_level_factory,
)
from minecombat.policies.harness.factory import (
    FixedTacticalPolicy,
)
from minecombat.policies.harness.high_level_session import (
    HighLevelRolloutSession,
)
from minecombat.policies.harness.tactical_policy import (
    HighLevelTacticalPolicy,
)


class FakeConditionedVPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.tactics_seen = []

    @property
    def device(self):
        return self.weight.device

    def get_action(
        self,
        obs,
        memory,
        *,
        tactical_condition,
        deterministic,
        input_shape,
    ):
        self.tactics_seen.append(tactical_condition.tactic_id)
        return {"attack": 1}, memory


def test_high_level_builder_freezes_h1(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fake")

    low_policy = FakeConditionedVPT()
    loader_calls = []

    def fake_h1_loader(**kwargs):
        loader_calls.append(kwargs)
        return HarnessRuntime(
            tactical_policy=FixedTacticalPolicy(0),
            low_policy=low_policy,
            gate=FixedIntervalGate(interval_steps=20),
        )

    monkeypatch.setattr(
        high_level_factory,
        "build_conditioned_harness",
        fake_h1_loader,
    )

    harness = (
        high_level_factory.build_high_level_training_harness(
            checkpoint,
            num_tactics=2,
            interval_steps=2,
            grid_size=4,
            hidden_dim=32,
        )
    )

    # H1 was loaded once with the expected checkpoint.
    assert len(loader_calls) == 1
    assert loader_calls[0]["conditioned_checkpoint"] == str(
        checkpoint
    )

    # Low-level policy must remain frozen.
    assert harness.low_policy is low_policy
    assert not low_policy.training
    assert all(
        not param.requires_grad
        for param in low_policy.parameters()
    )

    # The H2 policy must be trainable.
    high_policy = harness.tactical_policy

    assert isinstance(high_policy, HighLevelTacticalPolicy)
    assert high_policy.num_tactics == 2
    assert high_policy.feature_dim == 3 * 4 * 4

    assert all(
        param.requires_grad
        for param in high_policy.parameters()
    )

    assert harness.gate.interval_steps == 2

    # Verify the assembled runtime can collect a decision.
    session = HighLevelRolloutSession(harness)

    obs = {
        "image": np.zeros(
            (16, 16, 3),
            dtype=np.uint8,
        )
    }

    action = session.act(obs)

    assert action == {"attack": 1}
    assert low_policy.tactics_seen[0] in (0, 1)

    session.observe(
        obs,
        reward=1.0,
        truncated=True,
    )

    assert len(session.collector.transitions) == 1
    assert session.memory is None


def test_builder_requires_h1_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        high_level_factory.build_high_level_training_harness(
            tmp_path / "missing.ckpt"
        )
