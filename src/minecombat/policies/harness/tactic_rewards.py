
"""Tactic-dependent auxiliary rewards for low-level training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from minecombat.envs.callbacks.combat_stats import CombatStatsTracker


@dataclass(frozen=True)
class TacticalRewardConfig:
    """Auxiliary reward coefficients."""

    aggressive_damage_bonus: float = 0.005
    defensive_hp_penalty: float = 0.1

    def __post_init__(self):
        for value in (
            self.aggressive_damage_bonus,
            self.defensive_hp_penalty,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Reward coefficients must be finite and non-negative"
                )


class TacticalRewardCalculator:
    """Compute additional rewards for two tactical conditions.

    Tactic 0: Aggressive.
    Tactic 1: Defensive.
    """

    def __init__(self, config: TacticalRewardConfig | None = None):
        self.config = config or TacticalRewardConfig()
        self.tracker = CombatStatsTracker(target_mob="zombie")
        self._initialized = False

    def reset(self, info: Mapping) -> None:
        """Initialize counters for a new episode."""

        self.tracker.reset(info)
        self._initialized = True

    def compute(
        self,
        base_reward: float,
        info: Mapping,
        tactic_id: int,
    ) -> tuple[float, float]:
        """Return (training_reward, auxiliary_reward)."""

        if not self._initialized:
            raise RuntimeError("Call reset() before compute()")

        if type(tactic_id) is not int or tactic_id not in (0, 1):
            raise ValueError("Supported tactic IDs are 0 and 1")

        base_reward = float(base_reward)
        if not math.isfinite(base_reward):
            raise ValueError("Base reward must be finite")

        stats = self.tracker.update(info)

        if tactic_id == 0:
            auxiliary_reward = (
                self.config.aggressive_damage_bonus
                * stats.damage_dealt
            )
        else:
            auxiliary_reward = (
                -self.config.defensive_hp_penalty
                * stats.hp_lost
            )

        training_reward = base_reward + auxiliary_reward

        return training_reward, auxiliary_reward
