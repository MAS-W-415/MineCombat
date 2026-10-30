
"""Dense combat reward for the zombie arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._info import event_count, scalar

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class when MineStudio is not installed."""


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
    """Stateful reward calculator independent of MineStudio."""

    def __init__(
        self,
        config: CombatRewardConfig | None = None,
    ) -> None:
        self.config = (
            config if config is not None else CombatRewardConfig()
        )

        # Initialize historical state.
        self.reset({})

    def reset(self, info: dict[str, Any]) -> None:
        """Initialize numeric snapshots at the start of an episode."""

        target = self.config.target_mob

        self.prev_hp = scalar(
            info.get("health", 20.0),
            default=20.0,
        )

        self.prev_damage = event_count(
            info,
            "damage_dealt",
            target,
        )

        self.prev_kills = event_count(
            info,
            "kill_entity",
            target,
        )

        self._dead = self.prev_hp <= 0.0

    def compute(self, info: dict[str, Any]) -> float:
        """Calculate reward for one environment step."""

        cfg = self.config

        # 1. Read current cumulative event counters.
        damage_total = event_count(
            info,
            "damage_dealt",
            cfg.target_mob,
        )

        kills_total = event_count(
            info,
            "kill_entity",
            cfg.target_mob,
        )

        # 2. Calculate changes since the previous step.
        damage_dealt = max(
            0.0,
            damage_total - self.prev_damage,
        )

        kills = max(
            0.0,
            kills_total - self.prev_kills,
        )

        hp = scalar(
            info.get("health", self.prev_hp),
            default=self.prev_hp,
        )

        hp_lost = max(
            0.0,
            self.prev_hp - hp,
        )

        # 3. Apply the death penalty only once.
        death = 0.0

        if hp <= 0.0 and not self._dead:
            death = 1.0
            self._dead = True

        # 4. Calculate combat reward.
        reward = (
            cfg.kill * kills
            + cfg.damage_dealt * damage_dealt
            - cfg.damage_taken * hp_lost
            + cfg.death * death
            - cfg.time
        )

        # 5. Clip reward to the configured range.
        if cfg.clip_min is not None and cfg.clip_max is not None:
            reward = float(
                np.clip(reward, cfg.clip_min, cfg.clip_max)
            )

        # 6. Update numeric snapshots for the next step.
        self.prev_hp = hp
        self.prev_damage = damage_total
        self.prev_kills = kills_total

        return float(reward)


class CombatRewardCallback(MinecraftCallback):
    """Replace the simulator reward with the dense combat reward."""

    def __init__(
        self,
        config: CombatRewardConfig | dict[str, Any] | None = None,
    ) -> None:

        if isinstance(config, dict):
            config = CombatRewardConfig(**config)

        self.calculator = CombatRewardCalculator(config)

    def after_reset(self, sim, obs, info):
        self.calculator.reset(info)
        return obs, info

    def after_step(
        self,
        sim,
        obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        reward = self.calculator.compute(info)

        return obs, reward, terminated, truncated, info
