#!/usr/bin/env python3
"""Smoke-test the zombie arena with random actions or VPT 1x."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402


def _resolve(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(ROOT / candidate)


def _summarize_info(info: dict) -> dict:
    keys = ["health", "food_level", "kill_entity", "damage_dealt", "player_pos", "mobs"]
    summary = {}
    for key in keys:
        if key in info:
            value = info[key]
            if hasattr(value, "tolist"):
                value = value.tolist()
            summary[key] = value
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short zombie-arena smoke test.")
    parser.add_argument("--config", default="configs/env/zombie_stage1.yaml")
    parser.add_argument("--policy", choices=["random", "vpt"], default="random")
    parser.add_argument("--model", default="CraftJarvis/MineStudio_VPT.foundation_model_1x")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = make_zombie_env(_resolve(args.config))
    policy = None
    memory = None
    if args.policy == "vpt":
        from minecombat.policies import load_vpt_policy

        policy = load_vpt_policy(args.model)
        policy.eval()

    try:
        obs, info = env.reset()
        print("obs:", {key: getattr(value, "shape", None) for key, value in obs.items()})
        print("info keys:", sorted(info.keys()))
        print("reset info:", _summarize_info(info))

        started = time.perf_counter()
        for step in range(args.steps):
            if policy is None:
                action = env.action_space.sample()
            else:
                action, memory = policy.get_action(
                    obs,
                    memory,
                    deterministic=args.deterministic,
                    input_shape="*",
                )

            obs, reward, terminated, truncated, info = env.step(action)
            if step % 20 == 0 or terminated or truncated:
                print(
                    f"step={step:4d} reward={float(reward):+.3f} "
                    f"terminated={terminated} truncated={truncated} "
                    f"info={_summarize_info(info)}"
                )
            if terminated or truncated:
                break

        elapsed = time.perf_counter() - started
        print(f"steps={args.steps} elapsed={elapsed:.2f}s fps={args.steps / max(elapsed, 1e-6):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
