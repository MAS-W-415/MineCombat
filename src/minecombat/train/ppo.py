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

from .common import project_root
from .generators import (
    conditioned_env_generator,
    conditioned_policy_generator,
    env_generator,
    policy_generator,
)


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
        "--env-config",
        default=None,
        help="Override cfg.env.config, e.g. an alternate reward profile.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Warm-start flat PPO from a previous model.ckpt.",
    )
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-interval", type=int, default=None)
    parser.add_argument(
        "--record-video-interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny single-iteration profile for pipeline validation.",
    )
    parser.add_argument(
        "--conditioned",
        action="store_true",
        help="Enable conditioned VPT training.",
    )
    parser.add_argument(
        "--num-tactics",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--tactic-id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sample-tactic-per-episode",
        action="store_true",
    )
    parser.add_argument(
        "--tactical-rewards",
        action="store_true",
        help="Enable tactic-specific auxiliary rewards.",
    )
    args = parser.parse_args()

    for name in (
        "save_interval",
        "keep_interval",
        "record_video_interval",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.tactical_rewards:
        if not args.conditioned:
            parser.error(
                "--tactical-rewards requires --conditioned"
            )

        if args.num_tactics != 2:
            parser.error(
                "--tactical-rewards requires --num-tactics 2"
            )

    root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = OmegaConf.load(config_path)
    env_config = Path(str(cfg.env.config))
    if not env_config.is_absolute():
        cfg.env.config = str(root / env_config)

    if args.env_config is not None:
        env_config = Path(args.env_config)
        if not env_config.is_absolute():
            env_config = root / env_config
        if not env_config.is_file():
            parser.error(f"Environment config not found: {env_config}")
        cfg.env.config = str(env_config)

    init_checkpoint = None
    if args.init_checkpoint is not None:
        if args.conditioned:
            parser.error(
                "--init-checkpoint is only supported for flat training"
            )

        checkpoint_path = Path(args.init_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        if not checkpoint_path.is_file():
            parser.error(
                f"Initial checkpoint not found: {checkpoint_path}"
            )
        init_checkpoint = str(checkpoint_path)

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
    if args.save_interval is not None:
        cfg.train_config.save_interval = args.save_interval
    if args.keep_interval is not None:
        cfg.train_config.keep_interval = args.keep_interval
    if args.record_video_interval is not None:
        cfg.train_config.record_video_interval = (
            args.record_video_interval
        )
    if args.wandb_mode is not None:
        cfg.logger_config.mode = args.wandb_mode

    OmegaConf.update(
        cfg,
        "harness",
        {
            "conditioned": args.conditioned,
            "num_tactics": args.num_tactics,
            "tactic_id": args.tactic_id,
            "sample_tactic_per_episode": args.sample_tactic_per_episode,
            "tactical_rewards": args.tactical_rewards,
            "init_checkpoint": init_checkpoint,
        },
        force_add=True,
    )
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

    if args.conditioned:
        if args.num_tactics < 1:
            parser.error("--num-tactics must be positive")

        if not 0 <= args.tactic_id < args.num_tactics:
            parser.error("--tactic-id is out of range")

        env_gen = partial(
            conditioned_env_generator,
            cfg.env.config,
            args.num_tactics,
            args.tactic_id,
            args.sample_tactic_per_episode,
            args.tactical_rewards,
        )

        policy_gen = partial(
            conditioned_policy_generator,
            cfg.model.repo_id,
            args.num_tactics,
        )

    else:
        env_gen = partial(
            env_generator,
            cfg.env.config,
        )

        policy_gen = partial(
            policy_generator,
            cfg.model.repo_id,
            init_checkpoint,
        )
    online_cfg = build_online_config(cfg)
    whole_config = OmegaConf.to_yaml(cfg)
    start_rolloutmanager(policy_gen, env_gen, online_cfg)
    try:
        start_trainer(policy_gen, env_gen, online_cfg, whole_config)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
