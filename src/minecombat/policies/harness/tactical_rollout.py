
"""Semi-Markov rollout collection for high-level tactical RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .contracts import HarnessState, TacticalDecision


@dataclass(frozen=True)
class TacticalTransition:
    """One high-level transition spanning multiple environment steps."""

    features: Any
    tactic_id: int
    rewards: tuple[float, ...]
    next_features: Any
    terminated: bool
    truncated: bool

    # Behavior-policy statistics required by future high-level PPO.
    behavior_log_prob: float | None = None
    behavior_value: float | None = None

    @property
    def duration(self) -> int:
        return len(self.rewards)

    def discounted_reward(self, gamma: float) -> float:
        """Discount rewards within this tactical segment."""
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

        return sum(
            (gamma ** i) * reward
            for i, reward in enumerate(self.rewards)
        )

    def td_target(self, gamma: float, next_value: float) -> float:
        """Semi-Markov one-step TD target."""
        result = self.discounted_reward(gamma)

        # Genuine termination stops bootstrapping.
        # Time-limit truncation may still bootstrap.
        if not self.terminated:
            result += (gamma ** self.duration) * next_value

        return result


class TacticalRolloutCollector:
    """Collect rewards between consecutive high-level decisions.

    Usage order:
        on_action() -> record_reward() -> ...
        finish_episode() when the episode ends.

    Call on_action() after HarnessRuntime.get_action(), using
    the HarnessState returned for that action.
    """

    def __init__(self) -> None:
        self.transitions: list[TacticalTransition] = []
        self._active: dict[str, Any] | None = None

    def on_action(
        self,
        state: HarnessState,
        features: Any,
        *,
        behavior_log_prob: float | None = None,
        behavior_value: float | None = None,
    ) -> None:
        """Start a segment when Harness refreshes the tactic."""

        if state.last_decision_updated:
            if state.current_tactic is None:
                raise ValueError("Updated decision has no tactic")

            # A new high-level decision closes the old segment.
            if self._active is not None:
                self._close_segment(
                    next_features=features,
                    terminated=False,
                    truncated=False,
                )

            self._active = {
                "features": features,
                "tactic_id": state.current_tactic.tactic_id,
                "rewards": [],
                "behavior_log_prob": behavior_log_prob,
                "behavior_value": behavior_value,
            }

        elif self._active is None:
            raise RuntimeError(
                "No active tactical segment; expected first decision"
            )

        elif (
            state.current_tactic is None
            or state.current_tactic.tactic_id
            != self._active["tactic_id"]
        ):
            raise RuntimeError(
                "Tactic changed without a high-level decision update"
            )

    def record_reward(self, reward: float) -> None:
        """Record one primitive-step reward."""

        if self._active is None:
            raise RuntimeError("No active tactical segment")

        value = float(reward)

        if not math.isfinite(value):
            raise ValueError("Reward must be finite")

        self._active["rewards"].append(value)

    def finish_episode(
        self,
        next_features: Any,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Close the current segment at an episode boundary."""

        if not (terminated or truncated):
            raise ValueError(
                "Episode must be terminated or truncated"
            )

        self._close_segment(
            next_features=next_features,
            terminated=terminated,
            truncated=truncated,
        )

    def _close_segment(
        self,
        *,
        next_features: Any,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._active is None:
            raise RuntimeError("No active segment to close")

        if not self._active["rewards"]:
            raise RuntimeError(
                "Cannot close a segment before an environment step"
            )

        segment = self._active

        transition = TacticalTransition(
            features=segment["features"],
            tactic_id=segment["tactic_id"],
            rewards=tuple(segment["rewards"]),
            next_features=next_features,
            terminated=terminated,
            truncated=truncated,
            behavior_log_prob=segment["behavior_log_prob"],
            behavior_value=segment["behavior_value"],
        )

        self.transitions.append(transition)
        self._active = None

    def clear(self) -> None:
        """Clear collected data only between episodes."""

        if self._active is not None:
            raise RuntimeError(
                "Finish the active episode before clearing"
            )

        self.transitions.clear()
