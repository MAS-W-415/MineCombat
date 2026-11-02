
#!/usr/bin/env python3
"""Evaluate conditioned VPT under fixed tactical conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.eval import evaluate, summarize  # noqa: E402
from minecombat.policies.vpt import DEFAULT_MODEL  # noqa: E402
from minecombat.utils import RunLog  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed tactics with Conditioned VPT."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained Conditioned VPT model.ckpt",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--config",
        default="configs/env/zombie_stage1.yaml",
    )
    parser.add_argument("--num-tactics", type=int, default=2)
    parser.add_argument(
        "--tactic-ids",
        nargs="+",
        type=int,
        default=[0, 1],
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

    args = parser.parse_args()

    if args.num_tactics < 1:
        parser.error("--num-tactics must be positive")

    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")

    if any(
        tactic_id < 0 or tactic_id >= args.num_tactics
        for tactic_id in args.tactic_ids
    ):
        parser.error("tactic ID is outside the valid range")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()

    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    if not config.is_file():
        parser.error(f"Environment config not found: {config}")

    # Lazy imports: argument validation needs no Minecraft runtime.
    from minecombat.envs import make_zombie_env
    from minecombat.policies.harness.factory import (
        FixedTacticalPolicy,
        build_conditioned_harness,
    )

    # Load the trained model only once.
    harness = build_conditioned_harness(
        repo_id=args.model,
        device=args.device,
        conditioned_checkpoint=str(checkpoint),
        num_tactics=args.num_tactics,
        tactic_id=args.tactic_ids[0],
    )

    env = make_zombie_env(str(config))

    try:
        for tactic_id in args.tactic_ids:
            # Reuse the same low-level model weights.
            # Only the fixed high-level tactic changes.
            harness.tactical_policy = FixedTacticalPolicy(
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

            run = RunLog(
                args.output_dir,
                f"eval_conditioned_tactic_{tactic_id}",
            )

            run.save_config(str(config))
            run.save_metrics(
                {
                    "tactic_id": tactic_id,
                    "num_tactics": args.num_tactics,
                    "checkpoint": str(checkpoint),
                    "model": args.model,
                    "seed_start": args.seed_start,
                    "episodes": args.episodes,
                    "deterministic": args.deterministic,
                },
                filename="policy.json",
            )
            run.save_metrics(summary)
            run.save_records(records)
            run.save_git_commit()

            print(f"Saved results: {run.run_dir}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
