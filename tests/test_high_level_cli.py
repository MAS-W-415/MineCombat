
"""Tests for the HH2 training command-line entry point."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

import minecombat.envs as env_module
from minecombat.policies.harness import high_level_factory, high_level_trainer
from minecombat.train import high_level as cli


def test_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "high_level",
            "--h1-checkpoint", "missing.ckpt",
            "--num-updates", "3",
            "--dry-run",
        ],
    )

    cli.main()

    config = json.loads(capsys.readouterr().out)

    assert config["num_updates"] == 3
    assert config["num_tactics"] == 2
    assert config["dry_run"] is True


def test_invalid_interval(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "high_level",
            "--h1-checkpoint", "missing.ckpt",
            "--interval-steps", "0",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2


def test_cli_connects_training_components(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "h1.ckpt"
    checkpoint.write_bytes(b"fake")

    config = tmp_path / "arena.yaml"
    config.write_text("seed: 0\n")

    output = tmp_path / "h2_run"

    class FakeEnv:
        closed = False

        def close(self):
            self.closed = True

    env = FakeEnv()
    observed = {}

    policy = torch.nn.Linear(4, 2)

    harness = SimpleNamespace(
        tactical_policy=policy,
        feature_adapter=object(),
    )

    def fake_builder(**kwargs):
        observed["builder"] = kwargs
        return harness

    def fake_trainer(
        env_arg,
        session,
        optimizer,
        *,
        train_config,
        ppo_config,
        save_dir,
    ):
        observed["env"] = env_arg
        observed["session"] = session
        observed["optimizer"] = optimizer
        observed["train_config"] = train_config
        observed["save_dir"] = save_dir
        return []

    monkeypatch.setattr(
        high_level_factory,
        "build_high_level_training_harness",
        fake_builder,
    )
    monkeypatch.setattr(
        env_module,
        "make_zombie_env",
        lambda path: env,
    )
    monkeypatch.setattr(
        high_level_trainer,
        "train_high_level",
        fake_trainer,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "high_level",
            "--hh1-checkpoint", str(checkpoint),
            "--env-config", str(config),
            "--save-dir", str(output),
            "--num-updates", "2",
            "--episodes-per-update", "3",
            "--max-steps", "10",
            "--device", "cpu",
        ],
    )

    cli.main()

    assert observed["builder"]["conditioned_checkpoint"] == (
        checkpoint.resolve()
    )

    assert observed["env"] is env
    assert observed["session"].runtime is harness
    assert observed["train_config"].num_updates == 2
    assert observed["train_config"].episodes_per_update == 3
    assert observed["train_config"].max_steps == 10

    # Only high-level parameters go into the optimizer.
    optimized_ids = {
        id(param)
        for group in observed["optimizer"].param_groups
        for param in group["params"]
    }
    assert optimized_ids == {
        id(param) for param in policy.parameters()
    }

    assert observed["save_dir"] == output.resolve()
    assert env.closed

    run_config = json.loads(
        (output / "run_config.json").read_text()
    )
    assert run_config["num_updates"] == 2
