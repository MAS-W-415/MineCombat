"""Randomized zombie spawning for stage 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class used when MineStudio is not installed."""


@dataclass
class ZombieSpawnConfig:
    mob: str = "zombie"
    count: int = 1
    min_distance: float = 4.0
    max_distance: float = 10.0
    spawn_height: float = 2.0
    randomize_player_yaw: bool = True
    persistence_required: bool = True
    no_ai: bool = False
    # Optional curriculum controls. ``spawn_angle_deg=90`` places the
    # zombie at +Z, which is directly in front of a player at yaw 0.
    # ``spawn_angle_jitter_deg`` randomizes the angle around that base.
    spawn_angle_deg: float | None = None
    spawn_angle_jitter_deg: float | None = None
    player_yaw: float | None = None
    # Optional entity attributes, e.g. {"generic.movement_speed": 0.15}.
    attributes: dict[str, float] | None = None


class RandomZombieSpawnCallback(MinecraftCallback):
    """Spawn zombies on a random ring around the player after every reset."""

    def __init__(self, config: ZombieSpawnConfig | dict[str, Any] | None = None, **kwargs) -> None:
        if config is None:
            config = ZombieSpawnConfig(**kwargs)
        elif isinstance(config, dict):
            config = ZombieSpawnConfig(**config)
        self.config = config

    def after_reset(self, sim, obs, info):
        cfg = self.config
        rng = sim.np_random
        commands: list[str] = []

        if cfg.player_yaw is not None:
            commands.append(
                f"/tp @p ~ ~ ~ {float(cfg.player_yaw):.2f} 0.0"
            )
        elif cfg.randomize_player_yaw:
            yaw = float(rng.uniform(-180.0, 180.0))
            commands.append(f"/tp @p ~ ~ ~ {yaw:.2f} 0.0")

        tags = []
        if cfg.no_ai:
            tags.append("NoAI:1b")
        if cfg.persistence_required:
            tags.append("PersistenceRequired:1b")
        if cfg.attributes:
            attributes = ",".join(
                f'{{Name:"{name}",Base:{float(value)}d}}'
                for name, value in cfg.attributes.items()
            )
            tags.append(f"Attributes:[{attributes}]")
        nbt = "{" + ",".join(tags) + "}" if tags else ""

        spawns: list[dict[str, float | str]] = []
        for _ in range(int(cfg.count)):
            if cfg.spawn_angle_deg is not None:
                angle_deg = float(cfg.spawn_angle_deg)
                if cfg.spawn_angle_jitter_deg:
                    angle_deg += float(
                        rng.uniform(
                            -cfg.spawn_angle_jitter_deg,
                            cfg.spawn_angle_jitter_deg,
                        )
                    )
                angle = math.radians(angle_deg)
            else:
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
            distance = float(rng.uniform(cfg.min_distance, cfg.max_distance))
            x = math.cos(angle) * distance
            z = math.sin(angle) * distance
            spawns.append(
                {
                    "mob": cfg.mob,
                    "x": x,
                    "z": z,
                    "distance": distance,
                    "angle": angle,
                }
            )
            commands.append(
                f"/execute as @p at @p run summon minecraft:{cfg.mob} "
                f"~{x:.3f} ~{cfg.spawn_height:.3f} ~{z:.3f} {nbt}".rstrip()
            )

        if not commands:
            return obs, info

        for command in commands:
            obs, _, _, info = sim.env.execute_cmd(command)
        obs, info = sim._wrap_obs_info(obs, info)
        info["spawn_debug"] = spawns
        return obs, info
