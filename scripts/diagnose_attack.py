#!/usr/bin/env python3
"""Diagnose how often a policy presses attack / inventory.

This is an action-level diagnostic, not a performance evaluation. It runs the
policy deterministically in a chosen environment and counts the selected
button combinations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402
from minecombat.policies import load_vpt_policy  # noqa: E402


def resolve(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(ROOT / candidate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count attack and inventory actions for a checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="configs/env/zombie_stage1.yaml",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    env = make_zombie_env(resolve(args.config))
    policy = load_vpt_policy(
        checkpoint=resolve(args.checkpoint),
    )
    mapper = env.action_mapper

    counts: Counter[str] = Counter()
    attack_steps = 0
    inventory_steps = 0
    total_steps = 0
    outcomes: list[str | None] = []
    reward_sums: dict[str, float] = {}

    try:
        for _ in range(args.episodes):
            obs, info = env.reset()
            memory = None

            for _ in range(args.max_steps):
                action, memory = policy.get_action(
                    obs,
                    memory,
                    deterministic=True,
                    input_shape="*",
                )

                button_idx = int(
                    action["buttons"]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)[0]
                )
                combo = mapper.BUTTONS_IDX_TO_COMBINATION[button_idx]

                total_steps += 1
                if combo == "inventory":
                    inventory_steps += 1
                    counts["inventory"] += 1
                else:
                    if "attack" in combo:
                        attack_steps += 1
                        counts["attack"] += 1
                    counts[str(combo)] += 1

                obs, reward, terminated, truncated, info = env.step(action)
                breakdown = info.get("reward_breakdown")
                if isinstance(breakdown, dict):
                    for key, value in breakdown.items():
                        reward_sums[key] = reward_sums.get(key, 0.0) + float(value)
                if terminated or truncated:
                    break

            outcomes.append(info.get("combat_outcome"))
    finally:
        env.close()

    top = counts.most_common(10)
    summary = {
        "checkpoint": resolve(args.checkpoint),
        "config": resolve(args.config),
        "episodes": args.episodes,
        "total_steps": total_steps,
        "attack_steps": attack_steps,
        "attack_rate": attack_steps / max(1, total_steps),
        "inventory_steps": inventory_steps,
        "inventory_rate": inventory_steps / max(1, total_steps),
        "outcomes": outcomes,
        "reward_sums": {
            key: round(value, 4)
            for key, value in reward_sums.items()
        },
        "top_combinations": [
            {"combination": combo, "count": count}
            for combo, count in top
        ],
    }
    print(json.dumps(summary, indent=2))
    print(
        "RESULT",
        f"attack_rate={summary['attack_rate']:.4f}",
        f"inventory_rate={summary['inventory_rate']:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
