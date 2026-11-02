"""Shared cumulative combat statistics for reward calculators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._info import event_count, scalar


@dataclass(frozen=True)
class CombatStep:
    """Per-step deltas derived from cumulative Minecraft statistics."""

    hp: float
    hp_lost: float
    damage_dealt: float
    kills: float
    death: float


class CombatStatsTracker:
    """Track HP, damage, and kill deltas across environment steps."""

    def __init__(self, target_mob: str = "zombie") -> None:
        self.target_mob = target_mob
        self.reset({})

    @staticmethod
    def _damage_total(
        info: Mapping,
        target_mob: str,
    ) -> float:
        """Read damage from the top level or the custom-stat mapping."""

        value = info.get("damage_dealt")

        if value is None:
            custom = info.get("custom")
            if isinstance(custom, Mapping):
                value = custom.get("damage_dealt", 0.0)

        return event_count(
            {"damage_dealt": value},
            "damage_dealt",
            target_mob,
        )

    def reset(self, info: Mapping) -> None:
        """Snapshot cumulative counters at the start of an episode."""

        self.prev_hp = scalar(
            info.get("health", 20.0),
            default=20.0,
        )
        self.prev_damage = self._damage_total(
            info,
            self.target_mob,
        )
        self.prev_kills = event_count(
            info,
            "kill_entity",
            self.target_mob,
        )
        self._dead = self.prev_hp <= 0.0

    def update(self, info: Mapping) -> CombatStep:
        """Return this step's deltas and advance the snapshots."""

        damage_total = self._damage_total(
            info,
            self.target_mob,
        )
        kills_total = event_count(
            info,
            "kill_entity",
            self.target_mob,
        )

        hp = scalar(
            info.get("health", self.prev_hp),
            default=self.prev_hp,
        )

        damage_dealt = max(
            0.0,
            damage_total - self.prev_damage,
        )
        kills = max(
            0.0,
            kills_total - self.prev_kills,
        )
        hp_lost = max(
            0.0,
            self.prev_hp - hp,
        )

        death = 0.0
        if hp <= 0.0 and not self._dead:
            death = 1.0
            self._dead = True

        self.prev_hp = hp
        self.prev_damage = damage_total
        self.prev_kills = kills_total

        return CombatStep(
            hp=hp,
            hp_lost=hp_lost,
            damage_dealt=damage_dealt,
            kills=kills,
            death=death,
        )
