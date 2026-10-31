"""Tests for PPO preflight validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from minecombat.train.preflight import validate_ppo_config


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return OmegaConf.load(
        ROOT / "configs/ppo/zombie_vpt1x.yaml"
    )


def test_valid_configuration():
    cfg = load_config()

    total_gpu = validate_ppo_config(cfg, ROOT)

    assert total_gpu == pytest.approx(1.0)
    assert Path(cfg.env.config).is_absolute()
    assert Path(cfg.env.config).is_file()


def test_missing_environment_file():
    cfg = load_config()
    cfg.env.config = "configs/env/nonexistent.yaml"

    with pytest.raises(FileNotFoundError):
        validate_ppo_config(cfg, ROOT)


def test_gpu_budget_exceeded():
    cfg = load_config()
    cfg.train_config.num_gpus_per_worker = 0.75

    with pytest.raises(ValueError, match="budget exceeded"):
        validate_ppo_config(cfg, ROOT)


def test_invalid_worker_count():
    cfg = load_config()
    cfg.rollout_config.num_rollout_workers = 0

    with pytest.raises(ValueError, match="positive integer"):
        validate_ppo_config(cfg, ROOT)


def test_invalid_batch_size():
    cfg = load_config()
    cfg.rollout_config.worker_config.batch_size = 0

    with pytest.raises(ValueError, match="positive integer"):
        validate_ppo_config(cfg, ROOT)


def test_missing_model_name():
    cfg = load_config()
    cfg.model.repo_id = ""

    with pytest.raises(ValueError, match="model.repo_id"):
        validate_ppo_config(cfg, ROOT)