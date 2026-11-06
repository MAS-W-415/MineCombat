#!/usr/bin/env python3
"""Record evaluation episodes to MP4 files.

This is a deliverable helper, not part of the metrics path. It runs the same
policy/environment as ``scripts/eval.py`` and writes one video per episode plus
a small JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402
from minecombat.eval import seed_env  # noqa: E402
from minecombat.policies.vpt import load_vpt_policy  # noqa: E402


def resolve(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(ROOT / candidate)


def _bgr_frame(obs: dict, info: dict) -> np.ndarray:
    """Prefer the full-resolution POV frame, fall back to the policy image."""

    frame = info.get("pov")
    if frame is None:
        frame = obs["image"]
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    elif frame.shape[-1] == 3:
        # MineRL POV frames are RGB; OpenCV expects BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def _overlay(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = frame.copy()
    cv2.rectangle(canvas, (4, 4), (min(360, canvas.shape[1] - 4), 26), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Record VPT policy episodes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="configs/env/zombie_stage1.yaml",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="experiments/videos",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1 or args.fps < 1:
        parser.error("episodes, max-steps, and fps must be positive")

    policy = load_vpt_policy(
        checkpoint=resolve(args.checkpoint),
    )
    env = make_zombie_env(resolve(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []

    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            seed_env(env, seed)
            obs, info = env.reset()
            memory = None
            frames: list[np.ndarray] = [_bgr_frame(obs, info)]
            steps = 0
            terminated = truncated = False
            reward_sum = 0.0

            while steps < args.max_steps:
                action, memory = policy.get_action(
                    obs,
                    memory,
                    deterministic=args.deterministic,
                    input_shape="*",
                )
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                reward_sum += float(reward)
                frames.append(
                    _overlay(
                        _bgr_frame(obs, info),
                        f"seed {seed}  step {steps}",
                    )
                )
                if (
                    terminated
                    or truncated
                    or info.get("combat_outcome")
                    in {"success", "death", "timeout", "external_done"}
                ):
                    break

            height, width = frames[0].shape[:2]
            video_path = output_dir / f"episode_{seed:03d}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {video_path}")
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
            writer.release()

            outcome = info.get("combat_outcome", "unknown")
            record = {
                "seed": seed,
                "steps": steps,
                "reward": round(reward_sum, 4),
                "outcome": outcome,
                "video": str(video_path),
            }
            manifest.append(record)
            print(json.dumps(record), flush=True)
    finally:
        env.close()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved videos to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
