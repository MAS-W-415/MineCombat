#!/usr/bin/env python3
"""Evaluate a random or VPT policy on the zombie arena."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402
from minecombat.eval import evaluate, summarize  # noqa: E402
from minecombat.policies import DEFAULT_MODEL, load_vpt_policy  # noqa: E402
from minecombat.utils import RunLog  # noqa: E402


def _resolve(path: str) -> str:
    """Resolve paths relative to the project root."""

    candidate = Path(path)

    if candidate.is_absolute() or candidate.exists():
        return str(candidate)

    return str(ROOT / candidate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on the zombie arena."
    )

    parser.add_argument(
        "--config",
        default="configs/env/zombie_stage1.yaml",
    )

    parser.add_argument(
        "--policy",
        choices=["random", "vpt"],
        default="random",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional PPO model.ckpt to evaluate on top of VPT.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for VPT model inference.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--output-dir",
        default="experiments",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    args = parser.parse_args()

    if args.checkpoint and args.policy != "vpt":
        parser.error("--checkpoint requires --policy vpt")

    # Load the policy before creating the simulator.
    # A model-loading failure should not leave Minecraft running.
    policy = None

    if args.policy == "vpt":
        checkpoint = (
            _resolve(args.checkpoint)
            if args.checkpoint
            else None
        )
        print(
            f"Loading VPT model: {args.model}, "
            f"device={args.device}, checkpoint={checkpoint}"
        )

        policy = load_vpt_policy(
            repo_id=args.model,
            device=args.device,
            checkpoint=checkpoint,
        )

    print(
        f"Evaluating policy={args.policy}, "
        f"episodes={args.episodes}"
    )

    # The environment is created only after model loading.
    env = make_zombie_env(
        _resolve(args.config)
    )

    try:
        records = evaluate(
            env,
            policy=policy,
            episodes=args.episodes,
            seed_start=args.seed_start,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    summary = summarize(records)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    if not args.no_save:
        run_name = (
            "eval_vpt_ckpt"
            if args.checkpoint
            else f"eval_{args.policy}"
        )
        run = RunLog(
            args.output_dir,
            run_name,
        )

        run.save_config(
            _resolve(args.config)
        )

        run.save_metrics(
            {
                "policy": args.policy,
                "model": args.model,
                "checkpoint": (
                    _resolve(args.checkpoint)
                    if args.checkpoint
                    else None
                ),
            },
            filename="policy.json",
        )
        run.save_metrics(summary)
        run.save_records(records)
        run.save_git_commit()

        print(f"Saved results to: {run.run_dir}")


if __name__ == "__main__":
    main()