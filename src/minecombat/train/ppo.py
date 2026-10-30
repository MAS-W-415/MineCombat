"""PPO training entry point for the zombie arena.

This launcher adapts MineStudio's online training API to a single-GPU setup.
It is intentionally thin: the environment and reward logic live in
``minecombat.envs`` so the same code can be used for evaluation.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

from .generators import env_generator, policy_generator


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_online_config(cfg):
    return OmegaConf.create(
        {
            "trainer_name": cfg.trainer_name,
            "detach_rollout_manager": cfg.detach_rollout_manager,
            "rollout_config": OmegaConf.to_container(cfg.rollout_config, resolve=True),
            "train_config": OmegaConf.to_container(cfg.train_config, resolve=True),
            "logger_config": OmegaConf.to_container(cfg.logger_config, resolve=True),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VPT 1x on the zombie arena with PPO.")
    parser.add_argument(
        "--config",
        default="configs/ppo/zombie_vpt1x.yaml",
        help="Path to the PPO YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without starting Ray.",
    )
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--batches-per-iteration", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny single-iteration profile for pipeline validation.",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = OmegaConf.load(config_path)
    env_config = Path(str(cfg.env.config))
    if not env_config.is_absolute():
        cfg.env.config = str(root / env_config)

    if args.smoke:
        cfg.train_config.num_iterations = 1
        cfg.train_config.batches_per_iteration = 2
        cfg.train_config.gradient_accumulation = 1
        cfg.train_config.save_interval = 1
        cfg.train_config.record_video_interval = 1
        cfg.train_config.save_path = str(root / "experiments" / "ppo_smoke")
        cfg.rollout_config.worker_config.num_envs = 1
        cfg.rollout_config.worker_config.batch_size = 1
        cfg.logger_config.name = "zombie_vpt1x_smoke"
        cfg.logger_config.mode = "offline"

    if args.num_iterations is not None:
        cfg.train_config.num_iterations = args.num_iterations
    if args.batches_per_iteration is not None:
        cfg.train_config.batches_per_iteration = args.batches_per_iteration
    if args.num_envs is not None:
        cfg.rollout_config.worker_config.num_envs = args.num_envs
    if args.batch_size is not None:
        cfg.rollout_config.worker_config.batch_size = args.batch_size
    if args.save_path is not None:
        cfg.train_config.save_path = args.save_path
    if args.name is not None:
        cfg.logger_config.name = args.name

    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return

    import ray
    from minestudio.online.rollout.start_manager import start_rolloutmanager
    from minestudio.online.trainer.start_trainer import start_trainer

    ray.init(
        address="localhost:9899",
        ignore_reinit_error=True,
        namespace="online",
    )

    env_gen = partial(env_generator, cfg.env.config)
    policy_gen = partial(policy_generator, cfg.model.repo_id)

    online_cfg = build_online_config(cfg)
    whole_config = OmegaConf.to_yaml(cfg)
    start_rolloutmanager(policy_gen, env_gen, online_cfg)
    try:
        start_trainer(policy_gen, env_gen, online_cfg, whole_config)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
