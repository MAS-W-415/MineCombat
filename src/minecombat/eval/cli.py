"""Shared evaluation CLI for flat and conditioned policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..policies.vpt import DEFAULT_MODEL
from ..utils import RunLog
from . import evaluate, summarize


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str) -> str:
    """Resolve a path relative to the project root."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(project_root() / candidate)


def _save_run(
    *,
    output_dir: str,
    name: str,
    config: str,
    policy_meta: dict[str, Any],
    records: list[Any],
    summary: dict[str, Any],
) -> Path:
    run = RunLog(output_dir, name)
    run.save_config(config)
    run.save_metrics(policy_meta, filename="policy.json")
    run.save_metrics(summary)
    run.save_records(records)
    run.save_git_commit()
    return run.run_dir


def _run_flat(args: argparse.Namespace) -> int:
    from ..envs import make_zombie_env
    from ..policies.vpt import load_vpt_policy

    policy = None

    if args.policy == "vpt":
        checkpoint = (
            resolve_path(args.checkpoint)
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

    config = resolve_path(args.config)
    env = make_zombie_env(config)

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
    print(json.dumps(summary, indent=2))

    if not args.no_save:
        run_name = (
            "eval_vpt_ckpt"
            if args.checkpoint
            else f"eval_{args.policy}"
        )
        run_dir = _save_run(
            output_dir=args.output_dir,
            name=run_name,
            config=config,
            policy_meta={
                "policy": args.policy,
                "model": args.model,
                "checkpoint": (
                    resolve_path(args.checkpoint)
                    if args.checkpoint
                    else None
                ),
            },
            records=records,
            summary=summary,
        )
        print(f"Saved results to: {run_dir}")

    return 0


def _run_conditioned(args: argparse.Namespace) -> int:
    from .. import envs as env_module
    from ..policies.harness import factory

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()

    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise SystemExit(
            f"Environment config not found: {config}"
        )

    harness = factory.build_conditioned_harness(
        repo_id=args.model,
        device=args.device,
        conditioned_checkpoint=str(checkpoint),
        num_tactics=args.num_tactics,
        tactic_id=args.tactic_ids[0],
    )

    env = env_module.make_zombie_env(str(config))

    try:
        for tactic_id in args.tactic_ids:
            harness.tactical_policy = factory.FixedTacticalPolicy(
                tactic_id=tactic_id
            )

            records = evaluate(
                env,
                policy=harness,
                episodes=args.episodes,
                seed_start=args.seed_start,
                deterministic=args.deterministic,
                max_steps=args.max_steps,
            )

            summary = summarize(records)
            print(
                json.dumps(
                    {
                        "tactic_id": tactic_id,
                        "summary": summary,
                    },
                    indent=2,
                )
            )

            if not args.no_save:
                run_dir = _save_run(
                    output_dir=args.output_dir,
                    name=f"eval_conditioned_tactic_{tactic_id}",
                    config=str(config),
                    policy_meta={
                        "tactic_id": tactic_id,
                        "num_tactics": args.num_tactics,
                        "checkpoint": str(checkpoint),
                        "model": args.model,
                        "seed_start": args.seed_start,
                        "episodes": args.episodes,
                        "deterministic": args.deterministic,
                    },
                    records=records,
                    summary=summary,
                )
                print(f"Saved results: {run_dir}")
    finally:
        env.close()

    return 0


def build_parser(
    default_policy: str = "random",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate flat or conditioned policies."
    )
    parser.add_argument(
        "--policy",
        choices=["random", "vpt", "conditioned"],
        default=default_policy,
    )
    parser.add_argument(
        "--config",
        default="configs/env/zombie_stage1.yaml",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional flat PPO or required conditioned checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
    )
    parser.add_argument("--num-tactics", type=int, default=2)
    parser.add_argument(
        "--tactic-ids",
        nargs="+",
        type=int,
        default=[0, 1],
    )
    return parser


def main(
    argv: list[str] | None = None,
    default_policy: str = "random",
) -> int:
    parser = build_parser(default_policy)
    args = parser.parse_args(argv)

    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")

    if args.policy == "conditioned":
        if args.checkpoint is None:
            parser.error(
                "--checkpoint is required for --policy conditioned"
            )
        if args.num_tactics < 1:
            parser.error("--num-tactics must be positive")
        if any(
            tactic_id < 0 or tactic_id >= args.num_tactics
            for tactic_id in args.tactic_ids
        ):
            parser.error("tactic ID is outside the valid range")
        return _run_conditioned(args)

    if args.checkpoint and args.policy != "vpt":
        parser.error("--checkpoint requires --policy vpt")

    return _run_flat(args)
