
"""Tactic-dependent auxiliary rewards for low-level training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from minecombat.envs.callbacks._info import event_count, scalar


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
        self._initialized = False

    @staticmethod
    def _damage_total(info: Mapping) -> float:
        """Read the cumulative damage statistic."""

        value = info.get("damage_dealt")

        if value is None:
            custom = info.get("custom", {})
            if isinstance(custom, Mapping):
                value = custom.get("damage_dealt", 0.0)

        return event_count(
            {"damage_dealt": value},
            "damage_dealt",
            "zombie",
        )

    def reset(self, info: Mapping) -> None:
        """Initialize counters for a new episode."""

        self.prev_damage = self._damage_total(info)
        self.prev_hp = scalar(info.get("health"), default=20.0)
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

        damage_total = self._damage_total(info)
        hp = scalar(info.get("health"), default=self.prev_hp)

        damage_delta = max(
            0.0,
            damage_total - self.prev_damage,
        )

        hp_loss = max(
            0.0,
            self.prev_hp - hp,
        )

        if tactic_id == 0:
            auxiliary_reward = (
                self.config.aggressive_damage_bonus
                * damage_delta
            )
        else:
            auxiliary_reward = (
                -self.config.defensive_hp_penalty
                * hp_loss
            )

        # Update history exactly once per environment step.
        self.prev_damage = damage_total
        self.prev_hp = hp

        training_reward = base_reward + auxiliary_reward

        return training_reward, auxiliary_reward
