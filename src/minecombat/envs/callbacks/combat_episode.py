"""Manage combat episode outcomes and time limits."""

from __future__ import annotations

from typing import Any

from ._info import event_count, scalar

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class for local tests."""


class CombatEpisodeCallback(MinecraftCallback):
    """Determine success, death, timeout, or external termination."""

    def __init__(
        self,
        target_mob: str = "zombie",
        required_kills: int = 1,
        max_episode_steps: int = 600,
    ) -> None:
        if required_kills < 1:
            raise ValueError("required_kills must be at least 1")

        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        self.target_mob = target_mob
        self.required_kills = required_kills
        self.max_episode_steps = max_episode_steps

        self.episode_steps = 0
        self.initial_kills = 0.0

    def _add_outcome_info(
        self,
        info: dict[str, Any],
        outcome: str,
        episode_kills: float,
    ) -> dict[str, Any]:
        """Add combat metadata without modifying the input dict."""

        result = dict(info)

        result["combat_outcome"] = outcome
        result["combat_success"] = outcome == "success"
        result["combat_episode_steps"] = self.episode_steps
        result["combat_kills"] = episode_kills

        return result

    def after_reset(self, sim, obs, info):
        """Initialize state at the beginning of each episode."""

        self.episode_steps = 0

        self.initial_kills = event_count(
            info,
            "kill_entity",
            self.target_mob,
        )

        info = self._add_outcome_info(
            info,
            outcome="ongoing",
            episode_kills=0.0,
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
        """Evaluate the episode after every environment step."""

        self.episode_steps += 1

        current_kills = event_count(
            info,
            "kill_entity",
            self.target_mob,
        )

        episode_kills = max(
            0.0,
            current_kills - self.initial_kills,
        )

        hp = scalar(
            info.get("health", 20.0),
            default=20.0,
        )

        if hp <= 0.0:
            # Death takes precedence over simultaneous success.
            outcome = "death"
            terminated = True
            truncated = False

        elif episode_kills >= self.required_kills:
            outcome = "success"
            terminated = True
            truncated = False

        elif terminated or truncated:
            # Preserve termination signals from the underlying env.
            outcome = "external_done"

        elif self.episode_steps >= self.max_episode_steps:
            outcome = "timeout"
            terminated = False
            truncated = True

        else:
            outcome = "ongoing"

        info = self._add_outcome_info(
            info,
            outcome=outcome,
            episode_kills=episode_kills,
        )

        return obs, reward, terminated, truncated, info