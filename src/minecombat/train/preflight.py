"""Validate PPO configuration before starting Ray."""

from __future__ import annotations

import math
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def validate_ppo_config(
    cfg: DictConfig,
    root: Path,
) -> float:
    """Validate a single-GPU training profile.

    Returns the total Ray GPU resource allocation.
    """

    repo_id = OmegaConf.select(cfg, "model.repo_id")

    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("model.repo_id must be a non-empty string")

    env_config = OmegaConf.select(cfg, "env.config")

    if not isinstance(env_config, str) or not env_config.strip():
        raise ValueError("env.config must be a non-empty path")

    env_path = Path(env_config).expanduser()

    if not env_path.is_absolute():
        env_path = root / env_path

    env_path = env_path.resolve()

    if not env_path.is_file():
        raise FileNotFoundError(
            f"Environment config not found: {env_path}"
        )

    cfg.env.config = str(env_path)

    positive_integer_fields = (
        "rollout_config.num_rollout_workers",
        "rollout_config.worker_config.num_envs",
        "rollout_config.worker_config.batch_size",
        "rollout_config.fragment_length",
        "train_config.num_workers",
        "train_config.num_iterations",
        "train_config.batches_per_iteration",
        "train_config.gradient_accumulation",
        "train_config.batch_size_per_gpu",
        "train_config.context_length",
    )

    for field in positive_integer_fields:
        value = OmegaConf.select(cfg, field)

        if type(value) is not int or value < 1:
            raise ValueError(
                f"{field} must be a positive integer"
            )

    gpu_fields = (
        "rollout_config.num_gpus_per_worker",
        "train_config.num_gpus_per_worker",
    )

    for field in gpu_fields:
        value = OmegaConf.select(cfg, field)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"{field} must be a finite number between 0 and 1"
            )

    total_gpu = (
        cfg.rollout_config.num_rollout_workers
        * cfg.rollout_config.num_gpus_per_worker
        + cfg.train_config.num_workers
        * cfg.train_config.num_gpus_per_worker
    )

    if total_gpu > 1.0 + 1e-9:
        raise ValueError(
            f"Single-GPU budget exceeded: {total_gpu:.2f}"
        )

    return float(total_gpu)