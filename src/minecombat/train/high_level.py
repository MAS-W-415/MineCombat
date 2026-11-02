
"""Command-line entry point for H2 high-level PPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the H2 tactical Actor-Critic."
    )

    parser.add_argument(
        "--h1-checkpoint",
        required=True,
        help="Checkpoint of the trained H1 Conditioned VPT.",
    )
    parser.add_argument(
        "--env-config",
        default="configs/env/zombie_stage1.yaml",
    )
    parser.add_argument(
        "--save-dir",
        default="experiments/h2_smoke",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument("--num-tactics", type=int, default=2)
    parser.add_argument("--interval-steps", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--num-updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without loading VPT or Minecraft.",
    )

    args = parser.parse_args()

    # Validate arguments before any expensive imports.
    if args.num_tactics != 2:
        parser.error("Current H2 training requires 2 tactics")

    for name in (
        "interval_steps",
        "grid_size",
        "hidden_dim",
        "num_updates",
        "episodes_per_update",
        "max_steps",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be positive and finite")

    checkpoint = resolve_path(args.h1_checkpoint)
    env_config = resolve_path(args.env_config)
    save_dir = resolve_path(args.save_dir)

    run_config = {
        **vars(args),
        "h1_checkpoint": str(checkpoint),
        "env_config": str(env_config),
        "save_dir": str(save_dir),
    }

    if args.dry_run:
        print(json.dumps(run_config, indent=2))
        return

    if not checkpoint.is_file():
        parser.error(f"H1 checkpoint not found: {checkpoint}")

    if not env_config.is_file():
        parser.error(f"Environment config not found: {env_config}")

    if (save_dir / "high_level_latest.pt").exists():
        parser.error(
            "Checkpoint already exists in save-dir; "
            "resume training is not implemented yet"
        )

    # Heavy dependencies are only imported for real training.
    import random
    import numpy as np
    import torch

    from minecombat.envs import make_zombie_env
    from minecombat.policies.harness.high_level_factory import (
        build_high_level_training_harness,
    )
    from minecombat.policies.harness.high_level_session import (
        HighLevelRolloutSession,
    )
    from minecombat.policies.harness.high_level_trainer import (
        HighLevelTrainConfig,
        train_high_level,
    )
    from minecombat.policies.harness.high_level_ppo import (
        HighLevelPPOConfig,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load frozen H1 and construct trainable H2.
    harness = build_high_level_training_harness(
        conditioned_checkpoint=checkpoint,
        device=args.device,
        num_tactics=args.num_tactics,
        interval_steps=args.interval_steps,
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
    )

    session = HighLevelRolloutSession(harness)

    # Optimizer must contain H2 parameters only.
    optimizer = torch.optim.Adam(
        harness.tactical_policy.parameters(),
        lr=args.learning_rate,
    )

    # 2. Build the original combat environment.
    # Do not attach the H1 tactical reward wrapper.
    env = make_zombie_env(env_config)

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    try:
        # 3. Collect real episodes and update H2 PPO.
        train_high_level(
            env,
            session,
            optimizer,
            train_config=HighLevelTrainConfig(
                num_updates=args.num_updates,
                episodes_per_update=args.episodes_per_update,
                max_steps=args.max_steps,
            ),
            ppo_config=HighLevelPPOConfig(),
            save_dir=save_dir,
        )
    finally:
        env.close()

    print(f"H2 training complete: {save_dir}")


if __name__ == "__main__":
    main()
