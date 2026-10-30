#!/usr/bin/env python3
"""Validate 20 consecutive zombie-arena resets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402
from minecombat.eval import seed_env  # noqa: E402


def resolve(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(ROOT / candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate consecutive arena resets.")
    parser.add_argument("--config", default="configs/env/zombie_stage1.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--save-dir", default="experiments/m1_reset_validation")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    records = []

    env = make_zombie_env(resolve(args.config))
    try:
        for episode in range(args.episodes):
            seed_env(env, episode)
            obs, info = env.reset()
            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(env.noop_action())

            spawn = (info.get("spawn_debug") or [{}])[0]
            spawn_count = len(info.get("spawn_debug") or [])
            if spawn:
                # Minecraft yaw 0 faces +Z; atan2(-x, z) points at the spawn.
                yaw_target = float(np.degrees(np.arctan2(-spawn["x"], spawn["z"])))
                for _ in range(6):
                    env.env.execute_cmd(f"/tp @p ~ ~ ~ {yaw_target:.2f} 0.0")
                    obs, reward, terminated, truncated, info = env.step(env.noop_action())
                    if abs(float(info.get("player_pos", {}).get("yaw", 999)) - yaw_target) < 5:
                        break

            pos = info.get("player_pos", {})
            health = float(info.get("health", 0.0))
            record = {
                "episode": episode,
                "health": health,
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
                "z": float(pos.get("z", 0.0)),
                "yaw": float(pos.get("yaw", 0.0)),
                "spawn": spawn,
                "spawn_count": spawn_count,
            }
            records.append(record)
            frame = info["pov"].copy()
            cv2.putText(
                frame,
                f"ep {episode} hp {health:.0f} yaw {record['yaw']:.0f}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            frames.append(cv2.resize(frame, (320, 180)))
            print(record)

            if terminated or truncated:
                print(f"unexpected episode end at reset {episode}")
                return 1
    finally:
        env.close()

    rows = []
    for i in range(0, len(frames), 4):
        row = frames[i : i + 4]
        while len(row) < 4:
            row.append(np.zeros_like(frames[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    sheet_path = save_dir / "reset_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)

    healths = [r["health"] for r in records]
    spawn_counts = [r["spawn_count"] for r in records]
    yaws = {round(r["yaw"]) for r in records}
    positions = {(round(r["x"]), round(r["z"])) for r in records}
    passed = (
        all(h == 20.0 for h in healths)
        and all(c == 1 for c in spawn_counts)
        and len(yaws) >= 5
        and len(positions) >= 5
    )

    print(f"healths={healths}")
    print(f"spawn_counts={spawn_counts}")
    print(f"unique_yaws={len(yaws)} unique_positions={len(positions)}")
    print(f"contact_sheet={sheet_path}")
    print(f"RESULT passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
