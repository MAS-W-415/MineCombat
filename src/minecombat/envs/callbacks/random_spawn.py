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

        if cfg.randomize_player_yaw:
            yaw = float(rng.uniform(-180.0, 180.0))
            commands.append(f"/tp @p ~ ~ ~ {yaw:.2f} 0.0")

        nbt = "{PersistenceRequired:1b}" if cfg.persistence_required else ""
        for _ in range(int(cfg.count)):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            distance = float(rng.uniform(cfg.min_distance, cfg.max_distance))
            x = math.cos(angle) * distance
            z = math.sin(angle) * distance
            commands.append(
                f"/execute as @p at @p run summon minecraft:{cfg.mob} "
                f"~{x:.3f} ~{cfg.spawn_height:.3f} ~{z:.3f} {nbt}".rstrip()
            )

        if not commands:
            return obs, info

        for command in commands:
            obs, _, _, info = sim.env.execute_cmd(command)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
