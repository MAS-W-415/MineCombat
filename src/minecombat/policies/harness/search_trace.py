
"""Trace closed-loop search behavior without changing actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .visual_search_policy import (
    VisualSearchPolicy,
    VisualSearchState,
)


@dataclass
class EpisodeSearchTrace:
    steps: int = 0
    search_steps: int = 0
    align_steps: int = 0
    combat_steps: int = 0
    detected_steps: int = 0
    first_detection_step: int | None = None
    first_combat_step: int | None = None
    reacquisitions: int = 0

    _ever_detected: bool = False
    _previous_detected: bool = False

    def record(self, mode: str, detected: bool) -> None:
        self.steps += 1

        if mode == "search":
            self.search_steps += 1
        elif mode == "align":
            self.align_steps += 1
        elif mode == "combat":
            self.combat_steps += 1
            if self.first_combat_step is None:
                self.first_combat_step = self.steps
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        if detected:
            self.detected_steps += 1

            if self.first_detection_step is None:
                self.first_detection_step = self.steps

            if self._ever_detected and not self._previous_detected:
                self.reacquisitions += 1

            self._ever_detected = True

        self._previous_detected = detected

    def to_dict(self) -> dict[str, int | None]:
        return {
            "steps": self.steps,
            "search_steps": self.search_steps,
            "align_steps": self.align_steps,
            "combat_steps": self.combat_steps,
            "detected_steps": self.detected_steps,
            "first_detection_step": self.first_detection_step,
            "first_combat_step": self.first_combat_step,
            "reacquisitions": self.reacquisitions,
        }


class SearchTracePolicy:
    """Policy wrapper compatible with minecombat.eval.evaluate()."""

    def __init__(self, policy: VisualSearchPolicy):
        self.policy = policy
        self.episodes: list[EpisodeSearchTrace] = []

    def get_action(
        self,
        obs: dict,
        memory: VisualSearchState | None,
        *,
        deterministic: bool = True,
        input_shape: str = "*",
    ) -> tuple[dict, VisualSearchState]:
        action, next_memory = self.policy.get_action(
            obs,
            memory,
            deterministic=deterministic,
            input_shape=input_shape,
        )

        if memory is None:
            self.episodes.append(EpisodeSearchTrace())

        if not self.episodes:
            raise RuntimeError("No active search trace")

        detection = next_memory.last_detection

        detected = (
            detection is not None
            and detection.center_x is not None
            and detection.confidence
            >= self.policy.controller.config.min_confidence
        )

        self.episodes[-1].record(
            next_memory.mode,
            detected,
        )

        return action, next_memory

    def records(self) -> list[dict[str, Any]]:
        """Return JSON-serializable episode diagnostics."""
        return [
            trace.to_dict()
            for trace in self.episodes
        ]
