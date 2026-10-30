"""Stage-1 zombie arena built on MineStudio."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .callbacks import (
    ArenaResetCallback,
    CombatEpisodeCallback,
    CombatRewardCallback,
    CombatRewardConfig,
    RandomZombieSpawnCallback,
    ZombieSpawnConfig,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load environment configuration from YAML."""

    with Path(path).expanduser().open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle) or {}


def make_zombie_env(
    config: str | Path | Mapping[str, Any] | None = None,
    **overrides: Any,
):
    """Create the MineStudio zombie combat environment."""

    # 1. Resolve configuration.
    if config is None:
        cfg: dict[str, Any] = {}
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    elif isinstance(config, Mapping):
        cfg = dict(config)
    else:
        raise TypeError(
            f"Unsupported config type: {type(config)!r}"
        )

    cfg.update(overrides)

    # 2. Extract configuration sections.
    world = dict(cfg.get("world", {}))
    arena = dict(cfg.get("arena", {}))
    player = dict(cfg.get("player", {}))
    zombie = dict(cfg.get("zombie", {}))
    reward = dict(cfg.get("reward", {}))

    # Lazy imports keep local unit tests lightweight.
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback

    # 3. Construct callbacks.
    callbacks = [
        ArenaResetCallback(
            biome=world.get(
                "biome",
                cfg.get("preferred_spawn_biome", "plains"),
            ),
            random_tp_range=int(
                world.get("random_tp_range", 200)
            ),
            time=world.get("time", "midnight"),
            weather=world.get("weather", "clear"),
            do_mob_spawning=bool(
                world.get("do_mob_spawning", False)
            ),
            do_daylight_cycle=bool(
                world.get("do_daylight_cycle", False)
            ),
            difficulty=str(
                world.get("difficulty", "normal")
            ),
            build_platform=bool(
                arena.get("build_platform", True)
            ),
            platform_radius=int(
                arena.get("platform_radius", 8)
            ),
            night_vision=bool(
                arena.get("night_vision", True)
            ),
        ),

        CommandsCallback(
            list(player.get("commands", []))
        ),

        RandomZombieSpawnCallback(
            ZombieSpawnConfig(**zombie)
        ),

        CombatRewardCallback(
            CombatRewardConfig(**reward)
        ),

        CombatEpisodeCallback(
            target_mob=zombie.get("mob", "zombie"),
            required_kills=int(zombie.get("count", 1)),
            max_episode_steps=int(
                cfg.get("max_episode_steps", 600)
            ),
        ),
    ]

    # 4. Create the simulator.
    return MinecraftSim(
        obs_size=tuple(
            cfg.get("obs_size", (128, 128))
        ),
        render_size=tuple(
            cfg.get("render_size", (640, 360))
        ),
        seed=int(cfg.get("seed", 0)),
        preferred_spawn_biome=cfg.get(
            "preferred_spawn_biome",
            "plains",
        ),
        action_type=cfg.get("action_type", "agent"),
        callbacks=callbacks,
    )