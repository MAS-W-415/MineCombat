
"""Trace search behavior at episode and individual-step levels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .visual_search_policy import (
    VisualSearchPolicy,
    VisualSearchState,
)


def _action_index(value: Any) -> int:
    """Read a discrete agent action from NumPy or Torch."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)

    if array.size != 1:
        raise ValueError("Expected one discrete action index")

    return int(array.item())


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
    """Trace wrapper without changing underlying policy actions."""

    def __init__(self, policy: VisualSearchPolicy):
        self.policy = policy
        self.episodes: list[EpisodeSearchTrace] = []

        # One list of step records per episode.
        self.step_history: list[list[dict[str, Any]]] = []

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
            self.step_history.append([])

        if not self.episodes:
            raise RuntimeError("No active search trace")

        detection = next_memory.last_detection

        detected = (
            detection is not None
            and detection.center_x is not None
            and detection.confidence
            >= self.policy.controller.config.min_confidence
        )

        # Existing episode-level statistics.
        self.episodes[-1].record(
            next_memory.mode,
            detected,
        )

        # New step-level diagnostic record.
        self.step_history[-1].append({
            "step": next_memory.steps,
            "mode": next_memory.mode,
            "detected": bool(detected),
            "center_x": (
                detection.center_x
                if detection is not None
                else None
            ),
            "confidence": (
                detection.confidence
                if detection is not None
                else 0.0
            ),
            "buttons_id": _action_index(action["buttons"]),
            "camera_id": _action_index(action["camera"]),
        })

        return action, next_memory

    def records(self) -> list[dict[str, Any]]:
        """Existing episode summary interface."""
        return [
            trace.to_dict()
            for trace in self.episodes
        ]

    def export_steps(
        self,
        path: str | Path,
        *,
        seeds: list[int] | None = None,
    ) -> Path:
        """Export one JSON record per executed environment step."""

        if seeds is None:
            seeds = list(range(len(self.step_history)))

        if len(seeds) != len(self.step_history):
            raise ValueError(
                "Seed count must match episode count"
            )

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as handle:
            for episode_index, (seed, steps) in enumerate(
                zip(seeds, self.step_history, strict=True)
            ):
                for step_record in steps:
                    record = {
                        "episode_index": episode_index,
                        "seed": seed,
                        **step_record,
                    }

                    handle.write(
                        json.dumps(record) + "\n"
                    )

        return output
