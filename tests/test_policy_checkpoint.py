"""Tests for loading PPO checkpoints into the VPT policy."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
models = pytest.importorskip("minestudio.models")

from minecombat.policies.vpt import load_vpt_policy  # noqa: E402


class FakePolicy:
    def __init__(self) -> None:
        self.loaded = None
        self.strict = None
        self.device = None
        self.eval_called = False

    def load_state_dict(self, state, strict=True):
        self.loaded = state
        self.strict = strict
        return None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_checkpoint_is_loaded(monkeypatch, tmp_path):
    fake = FakePolicy()

    class FakeVPTPolicy:
        @staticmethod
        def from_pretrained(repo_id):
            assert repo_id == "fake/model"
            return fake

    monkeypatch.setattr(models, "VPTPolicy", FakeVPTPolicy)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"weight": torch.ones(1)}, checkpoint)

    policy = load_vpt_policy(
        repo_id="fake/model",
        device="cpu",
        checkpoint=str(checkpoint),
    )

    assert policy is fake
    assert fake.loaded["weight"].item() == 1.0
    assert fake.strict is True
    assert fake.device == "cpu"
    assert fake.eval_called is True
