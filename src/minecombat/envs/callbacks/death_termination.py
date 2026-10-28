"""Terminate an episode when the player dies."""

from __future__ import annotations

from ._info import scalar

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class used when MineStudio is not installed."""


class DeathTerminationCallback(MinecraftCallback):
    """Set ``terminated`` when the player's health reaches zero."""

    def __init__(self, health_key: str = "health", threshold: float = 0.0) -> None:
        self.health_key = health_key
        self.threshold = threshold
        self.dead = False

    def after_reset(self, sim, obs, info):
        self.dead = scalar(info.get(self.health_key, 20.0)) <= self.threshold
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        hp = scalar(info.get(self.health_key, 20.0))
        if hp <= self.threshold:
            terminated = True
            self.dead = True
        return obs, reward, terminated, truncated, info
