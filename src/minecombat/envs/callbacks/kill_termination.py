
"""Terminate a combat episode after the required target kills."""

from __future__ import annotations

from ._info import event_count

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class for tests without MineStudio."""


class KillTerminationCallback(MinecraftCallback):
    """Terminate the episode when enough target mobs are killed."""

    def __init__(
        self,
        target_mob: str = "zombie",
        required_kills: int = 1,
    ) -> None:
        if required_kills < 1:
            raise ValueError("required_kills must be at least 1")

        self.target_mob = target_mob
        self.required_kills = required_kills
        self.initial_kills = 0.0

    def after_reset(self, sim, obs, info):
        """Record the kill counter at the beginning of an episode."""
        self.initial_kills = event_count(
            info,
            "kill_entity",
            self.target_mob,
        )

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
        """Check whether the current episode has met its kill goal."""
        current_kills = event_count(
            info,
            "kill_entity",
            self.target_mob,
        )

        episode_kills = max(
            0.0,
            current_kills - self.initial_kills,
        )

        if episode_kills >= self.required_kills:
            terminated = True

        return obs, reward, terminated, truncated, info
