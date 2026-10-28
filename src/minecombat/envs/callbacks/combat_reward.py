"""Dense combat reward for the zombie arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._info import event_count, scalar

try:  # Keep the reward math importable without a MineStudio installation.
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover - exercised only in lightweight test envs
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class used when MineStudio is not installed."""


@dataclass
class CombatRewardConfig:
    """Weights for the stage-1 combat reward."""

    target_mob: str = "zombie"
    kill: float = 10.0
    death: float = -5.0
    damage_dealt: float = 0.1
    damage_taken: float = 0.2
    time: float = 0.001
    clip_min: float = -10.0
    clip_max: float = 20.0


class CombatRewardCalculator:
    """Pure reward calculator, independent of the simulator.

    Keeping this class separate makes the reward easy to unit test and easy to
    reuse in offline analysis.
    """

    def __init__(self, config: CombatRewardConfig | None = None) -> None:
        self.config = config or CombatRewardConfig()
        self.prev_info: dict[str, Any] = {}
        self._dead = False

    def reset(self, info: dict[str, Any]) -> None:
        self.prev_info = dict(info)
        self._dead = scalar(info.get("health", 20.0)) <= 0.0

    def compute(self, info: dict[str, Any]) -> float:
        cfg = self.config
        prev = self.prev_info

        damage_dealt = max(
            0.0,
            event_count(info, "damage_dealt", cfg.target_mob)
            - event_count(prev, "damage_dealt", cfg.target_mob),
        )
        kills = max(
            0.0,
            event_count(info, "kill_entity", cfg.target_mob)
            - event_count(prev, "kill_entity", cfg.target_mob),
        )

        prev_hp = scalar(prev.get("health", 20.0), default=20.0)
        hp = scalar(info.get("health", prev_hp), default=prev_hp)
        hp_lost = max(0.0, prev_hp - hp)

        death = 0.0
        if hp <= 0.0 and not self._dead:
            death = 1.0
            self._dead = True

        reward = (
            cfg.kill * kills
            + cfg.damage_dealt * damage_dealt
            - cfg.damage_taken * hp_lost
            + cfg.death * death
            - cfg.time
        )
        if cfg.clip_min is not None and cfg.clip_max is not None:
            reward = float(np.clip(reward, cfg.clip_min, cfg.clip_max))

        self.prev_info = dict(info)
        return float(reward)


class CombatRewardCallback(MinecraftCallback):
    """Replace the simulator reward with the dense combat reward."""

    def __init__(self, config: CombatRewardConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, dict):
            config = CombatRewardConfig(**config)
        self.calculator = CombatRewardCalculator(config)

    def after_reset(self, sim, obs, info):
        self.calculator.reset(info)
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        reward = self.calculator.compute(info)
        return obs, reward, terminated, truncated, info
