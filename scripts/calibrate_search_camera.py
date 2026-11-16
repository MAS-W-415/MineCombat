
#!/usr/bin/env python3
"""Calibrate MineStudio camera direction using real game frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.envs import make_zombie_env  # noqa: E402
from minecombat.envs.zombie_arena import load_config  # noqa: E402
from minecombat.policies.harness.visual_search_policy import (  # noqa: E402
    CameraActionAdapter,
)


def save_frame(path, obs):
    """Save the actual RGB observation received by VPT."""
    import cv2

    rgb = np.asarray(obs["image"])
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected RGB observation")

    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Cannot write {path}")


def check_action(env, action, requested_yaw):
    """Check discrete indices, decode and reject extra buttons."""

    buttons = np.asarray(action["buttons"])
    camera = np.asarray(action["camera"])

    if buttons.size != 1 or camera.size != 1:
        raise ValueError("Expected one unbatched agent action")

    button_id = int(buttons.reshape(-1)[0])
    camera_id = int(camera.reshape(-1)[0])

    if not 0 <= button_id < 8641:
        raise ValueError("Invalid button action ID")
    if not 0 <= camera_id < 121:
        raise ValueError("Invalid camera action ID")

    decoded = env.agent_action_to_env_action({
        "buttons": buttons.copy(),
        "camera": camera.copy(),
    })

    extra_buttons = [
        key for key, value in decoded.items()
        if key != "camera" and np.any(np.asarray(value) != 0)
    ]

    if extra_buttons:
        raise RuntimeError(
            f"Unexpected buttons activated: {extra_buttons}"
        )

    # MineStudio camera order: pitch, yaw.
    camera_delta = np.asarray(decoded["camera"]).reshape(-1)

    if camera_delta.size != 2:
        raise RuntimeError("Invalid decoded camera format")

    applied_yaw = float(camera_delta[1])

    if applied_yaw * requested_yaw <= 0:
        raise RuntimeError(
            f"Unexpected yaw: requested={requested_yaw}, "
            f"decoded={applied_yaw}"
        )

    return {
        "buttons_id": button_id,
        "camera_id": camera_id,
        "decoded_pitch": float(camera_delta[0]),
        "decoded_yaw": applied_yaw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/env/zombie_stage1_random_face.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/search_camera_calibration",
    )
    parser.add_argument("--degrees", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not 0 < args.degrees <= 10:
        parser.error("degrees must be between 0 and 10")
    if args.steps < 1:
        parser.error("steps must be positive")

    output = Path(args.output_dir)
    if output.exists():
        parser.error(f"Output directory already exists: {output}")

    cfg = load_config(args.config)

    # Calibration only: keep the Zombie stationary and initially visible.
    cfg["zombie"] = {
        **cfg["zombie"],
        "face_spawn": True,
        "no_ai": True,
        "spawn_height": 0.0,
        "min_distance": 5.0,
        "max_distance": 6.0,
    }
    cfg["assist"] = {}

    output.mkdir(parents=True)
    env = make_zombie_env(cfg)

    results = []

    try:
        adapter = CameraActionAdapter(env, yaw_sign=1)

        for name, yaw in (
            ("positive", args.degrees),
            ("negative", -args.degrees),
        ):
            obs, _ = env.reset()

            # Refresh frames after teleport/spawn commands.
            for _ in range(2):
                obs, _, terminated, truncated, _ = env.step(
                    env.noop_action()
                )
                if terminated or truncated:
                    raise RuntimeError("Calibration ended early")

            save_frame(output / f"before_{name}.png", obs)

            action_info = None

            for _ in range(args.steps):
                action = adapter(yaw)
                action_info = check_action(env, action, yaw)

                # MinecraftSim.step mutates its action dictionary.
                safe_action = {
                    key: np.asarray(value).copy()
                    for key, value in action.items()
                }

                obs, _, terminated, truncated, _ = env.step(
                    safe_action
                )

                if terminated or truncated:
                    raise RuntimeError("Calibration ended early")

            save_frame(output / f"after_{name}.png", obs)

            results.append({
                "direction": name,
                "requested_yaw": yaw,
                "steps": args.steps,
                **action_info,
            })

    finally:
        env.close()

    (output / "calibration.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Saved calibration images: {output}")


if __name__ == "__main__":
    main()
