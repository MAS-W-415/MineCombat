#!/usr/bin/env python3
"""Validate timeout and death outcome logic.

Timeout is tested against the live simulator. Death is tested against the
callback logic because Malmo's zombie AI makes a natural death flaky; the
callback unit tests cover the same path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402
from minecombat.envs.callbacks import CombatEpisodeCallback  # noqa: E402


def resolve(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(ROOT / candidate)


def test_timeout(env) -> bool:
    env.reset()
    env.env.execute_cmd("/kill @e[type=minecraft:zombie]")
    info = None
    truncated = False
    for _ in range(600):
        _, _, terminated, truncated, info = env.step(env.noop_action())
        if terminated or truncated:
            break
    outcome = (info or {}).get("combat_outcome")
    print("timeout:", "truncated=", truncated, "outcome=", outcome)
    return bool(truncated and outcome == "timeout")


def test_death_logic() -> bool:
    callback = CombatEpisodeCallback(
        target_mob="zombie",
        required_kills=1,
        max_episode_steps=600,
    )
    info = {"health": 20.0, "kill_entity": {}, "damage_dealt": {}}
    callback.after_reset(None, {}, info)
    _, _, terminated, truncated, info = callback.after_step(
        None,
        {},
        0.0,
        False,
        False,
        {"health": 0.0, "kill_entity": {}, "damage_dealt": {}},
    )
    outcome = info.get("combat_outcome")
    print("death logic:", "terminated=", terminated, "outcome=", outcome)
    return bool(terminated and not truncated and outcome == "death")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate timeout and death outcomes.")
    parser.add_argument("--config", default="configs/env/zombie_stage1.yaml")
    args = parser.parse_args()

    env = make_zombie_env(resolve(args.config))
    try:
        timeout_ok = test_timeout(env)
    finally:
        env.close()

    death_ok = test_death_logic()
    print(f"RESULT timeout={timeout_ok} death_logic={death_ok}")
    return 0 if timeout_ok and death_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
