"""Tests for VPT model loading."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from minecombat.policies import DEFAULT_MODEL, load_vpt_policy


class FakeVPTPolicy:
    """A lightweight replacement for MineStudio's VPTPolicy."""

    def __init__(self, repo_id):
        self.repo_id = repo_id
        self.device = None
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, repo_id):
        return cls(repo_id)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def install_fake_dependencies(monkeypatch, cuda_available):
    """Provide fake torch and MineStudio modules."""

    torch_module = ModuleType("torch")

    torch_module.cuda = SimpleNamespace(
        is_available=lambda: cuda_available
    )

    minestudio_module = ModuleType("minestudio")
    minestudio_module.__path__ = []

    models_module = ModuleType("minestudio.models")
    models_module.VPTPolicy = FakeVPTPolicy

    monkeypatch.setitem(
        sys.modules, "torch", torch_module
    )

    monkeypatch.setitem(
        sys.modules, "minestudio", minestudio_module
    )

    monkeypatch.setitem(
        sys.modules, "minestudio.models", models_module
    )


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [
        (True, "cuda"),
        (False, "cpu"),
    ],
)
def test_auto_device_selection(
    monkeypatch,
    cuda_available,
    expected_device,
):
    install_fake_dependencies(
        monkeypatch,
        cuda_available=cuda_available,
    )

    policy = load_vpt_policy()

    assert policy.repo_id == DEFAULT_MODEL
    assert policy.device == expected_device
    assert policy.eval_called is True


def test_explicit_cpu_and_custom_model(monkeypatch):
    install_fake_dependencies(
        monkeypatch,
        cuda_available=True,
    )

    policy = load_vpt_policy(
        repo_id="example/custom-vpt",
        device="cpu",
    )

    assert policy.repo_id == "example/custom-vpt"
    assert policy.device == "cpu"
    assert policy.eval_called is True


def test_unavailable_cuda_raises(monkeypatch):
    install_fake_dependencies(
        monkeypatch,
        cuda_available=False,
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        load_vpt_policy(device="cuda")