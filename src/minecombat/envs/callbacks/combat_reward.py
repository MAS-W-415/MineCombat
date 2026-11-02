
"""Dense combat reward for the zombie arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._info import scalar
from .combat_stats import CombatStatsTracker

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
    # Minecraft stores damage_dealt in tenths of a HP.
    damage_dealt: float = 0.01
    damage_taken: float = 0.2
    time: float = 0.001
    # Optional attack-attempt shaping for action discovery.
    attack: float = 0.0
    attack_cooldown_steps: int = 10
    clip_min: float = -10.0
    clip_max: float = 20.0

    def __post_init__(self) -> None:
        if self.attack < 0:
            raise ValueError("attack reward must be non-negative")
        if self.attack_cooldown_steps < 1:
            raise ValueError(
                "attack_cooldown_steps must be positive"
            )


class CombatRewardCalculator:
    """Stateful reward calculator independent of MineStudio."""

    def __init__(
        self,
        config: CombatRewardConfig | None = None,
    ) -> None:
        self.config = (
            config if config is not None else CombatRewardConfig()
        )
        self.tracker = CombatStatsTracker(self.config.target_mob)

        # Initialize historical state.
        self.reset({})

    def reset(self, info: dict[str, Any]) -> None:
        """Initialize numeric snapshots at the start of an episode."""

        self.tracker.reset(info)
        self._attack_cooldown = 0

    def compute(
        self,
        info: dict[str, Any],
        attack_pressed: bool = False,
    ) -> float:
        """Calculate reward for one environment step."""

        cfg = self.config

        # 1. Read this step's HP, damage, kill, and death deltas.
        stats = self.tracker.update(info)

        # 2. Reward attack attempts, with a cooldown to avoid spam.
        attack_bonus = 0.0
        if attack_pressed and self._attack_cooldown == 0:
            attack_bonus = cfg.attack
            self._attack_cooldown = cfg.attack_cooldown_steps
        elif self._attack_cooldown > 0:
            self._attack_cooldown -= 1

        # 3. Calculate combat reward.
        reward = (
            cfg.kill * stats.kills
            + cfg.damage_dealt * stats.damage_dealt
            - cfg.damage_taken * stats.hp_lost
            + cfg.death * stats.death
            + attack_bonus
            - cfg.time
        )

        # 4. Clip reward to the configured range.
        if cfg.clip_min is not None and cfg.clip_max is not None:
            reward = float(
                np.clip(reward, cfg.clip_min, cfg.clip_max)
            )

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
        self._attack_pressed = False

    def before_step(self, sim, action):
        self._attack_pressed = (
            float(scalar(action.get("attack", 0), default=0.0))
            > 0.5
        )
        return action

    def after_reset(self, sim, obs, info):
        self.calculator.reset(info)
        self._attack_pressed = False
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
        reward = self.calculator.compute(
            info,
            attack_pressed=self._attack_pressed,
        )

        return obs, reward, terminated, truncated, info
