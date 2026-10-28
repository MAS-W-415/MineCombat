"""Stage-1 zombie arena built on MineStudio."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .callbacks import (
    ArenaResetCallback,
    CombatRewardCallback,
    CombatRewardConfig,
    DeathTerminationCallback,
    RandomZombieSpawnCallback,
    ZombieSpawnConfig,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_zombie_env(config: str | Path | Mapping[str, Any] | None = None, **overrides: Any):
    """Create a MineStudio environment for the stage-1 zombie task.

    Parameters
    ----------
    config:
        Path to a YAML config, a config mapping, or ``None`` to use defaults.
    **overrides:
        Top-level config overrides.
    """
    if config is None:
        cfg: dict[str, Any] = {}
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    elif isinstance(config, Mapping):
        cfg = dict(config)
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")
    cfg.update(overrides)

    world = dict(cfg.get("world", {}))
    player = dict(cfg.get("player", {}))
    zombie = dict(cfg.get("zombie", {}))
    reward = dict(cfg.get("reward", {}))

    # Imported lazily so unit tests can import minecombat without MineStudio.
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback, JudgeResetCallback

    callbacks = [
        ArenaResetCallback(
            biome=world.get("biome", cfg.get("preferred_spawn_biome", "plains")),
            random_tp_range=int(world.get("random_tp_range", 200)),
            time=world.get("time", "midnight"),
            weather=world.get("weather", "clear"),
            do_mob_spawning=bool(world.get("do_mob_spawning", False)),
            do_daylight_cycle=bool(world.get("do_daylight_cycle", False)),
        ),
        CommandsCallback(list(player.get("commands", []))),
        RandomZombieSpawnCallback(ZombieSpawnConfig(**zombie)),
        CombatRewardCallback(CombatRewardConfig(**reward)),
        DeathTerminationCallback(),
        JudgeResetCallback(int(cfg.get("max_episode_steps", 600))),
    ]

    return MinecraftSim(
        obs_size=tuple(cfg.get("obs_size", (128, 128))),
        render_size=tuple(cfg.get("render_size", (640, 360))),
        seed=int(cfg.get("seed", 0)),
        preferred_spawn_biome=cfg.get("preferred_spawn_biome", "plains"),
        action_type=cfg.get("action_type", "agent"),
        callbacks=callbacks,
    )
