#!/usr/bin/env python3
"""Scripted melee integration test for the zombie arena.

The test builds a flat platform (via the arena reset callback), faces the
player, summons a stationary zombie in front, and attacks with the native
agent action space. It checks damage, kill, reward, and episode outcome.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402


def resolve(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(ROOT / candidate)


def make_attack_action(env):
    mapper = env.action_mapper
    combo = tuple("attack" if key == "attack" else "none" for key in mapper.BUTTONS_GROUPS.keys())
    attack_idx = mapper.BUTTONS_COMBINATION_TO_IDX[combo]
    camera_idx = mapper.camera_null_idx
    return {"buttons": np.array([[attack_idx]]), "camera": np.array([[camera_idx]])}


def face_zero_yaw(env, attempts: int = 6):
    obs = info = None
    for _ in range(attempts):
        env.env.execute_cmd("/tp @p ~ ~ ~ 0.0 0.0")
        obs, reward, terminated, truncated, info = env.step(env.noop_action())
        yaw = float(info.get("player_pos", {}).get("yaw", 999.0))
        if abs(yaw) < 5.0:
            return obs, info
    return obs, info


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a scripted zombie melee test.")
    parser.add_argument("--config", default="configs/env/zombie_stage1.yaml")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--save-dir", default="experiments/m1_melee_test")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_zombie_env(resolve(args.config))
    try:
        obs, info = env.reset()
        env.env.execute_cmd("/kill @e[type=minecraft:zombie]")
        obs, info = face_zero_yaw(env)

        env.env.execute_cmd(
            f"/execute as @p at @p run summon minecraft:zombie ^ ^ ^{args.distance} "
            "{NoAI:1b,PersistenceRequired:1b}"
        )
        # Let the entity settle before the first attack.
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.noop_action())
        cv2.imwrite(str(save_dir / "before_attack.png"), info["pov"])

        print("player:", info.get("player_pos"))
        print("health:", info.get("health"))
        print("outcome:", info.get("combat_outcome"))

        total_reward = 0.0
        max_damage = 0.0
        max_kills = 0.0
        final_info = info
        final_terminated = False

        for step in range(args.steps):
            # Alternate press/release: MineStudio's key command is edge-triggered.
            action = make_attack_action(env) if step % 2 == 0 else env.noop_action()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            final_info = info
            final_terminated = terminated or truncated

            damage = dict(info.get("damage_dealt") or {})
            kills = dict(info.get("kill_entity") or {})
            max_damage = max(max_damage, float(damage.get("zombie", 0.0)))
            max_kills = max(max_kills, float(kills.get("zombie", 0.0)))

            if step % 10 == 0 or max_damage > 0 or final_terminated:
                cv2.imwrite(str(save_dir / f"step_{step:03d}.png"), info["pov"])
                print(
                    f"step={step:3d} reward={float(reward):+.3f} "
                    f"health={info.get('health')} damage={damage} kills={kills} "
                    f"outcome={info.get('combat_outcome')} terminated={terminated}"
                )

            if final_terminated:
                break

        cv2.imwrite(str(save_dir / "after_attack.png"), final_info["pov"])
        outcome = final_info.get("combat_outcome")
        passed = max_damage > 0 and max_kills >= 1 and outcome == "success"
        print(
            f"RESULT damage={max_damage} kills={max_kills} outcome={outcome} "
            f"reward={total_reward:.3f} passed={passed}"
        )
        return 0 if passed else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
