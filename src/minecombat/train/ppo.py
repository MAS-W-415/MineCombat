"""PPO training entry point for the zombie arena.

This launcher adapts MineStudio's online training API to a single-GPU setup.
It is intentionally thin: the environment and reward logic live in
``minecombat.envs`` so the same code can be used for evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


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
    args = parser.parse_args()

    root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = OmegaConf.load(config_path)
    env_config = Path(str(cfg.env.config))
    if not env_config.is_absolute():
        cfg.env.config = str(root / env_config)

    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return

    import ray
    from minestudio.online.rollout.start_manager import start_rolloutmanager
    from minestudio.online.trainer.start_trainer import start_trainer

    from minecombat.envs import make_zombie_env

    ray.init(ignore_reinit_error=True)

    def env_generator():
        return make_zombie_env(cfg.env.config)

    def policy_generator():
        from minestudio.models import VPTPolicy
        from minestudio.utils import get_compute_device

        return VPTPolicy.from_pretrained(cfg.model.repo_id).to(get_compute_device())

    online_cfg = build_online_config(cfg)
    start_rolloutmanager(policy_generator, env_generator, online_cfg)
    start_trainer(policy_generator, env_generator, online_cfg)


if __name__ == "__main__":
    main()
