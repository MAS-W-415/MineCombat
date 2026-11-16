
"""Observation-driven search and target alignment controller."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class TargetDetection:
    # None means that no target has been localized.
    center_x: float | None = None
    confidence: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid detection confidence")

        if self.center_x is not None:
            if not math.isfinite(self.center_x) or not -1 <= self.center_x <= 1:
                raise ValueError("center_x must be in [-1, 1]")


@dataclass(frozen=True)
class SearchConfig:
    min_confidence: float = 0.5
    center_tolerance: float = 0.15
    scan_yaw: float = 10.0
    alignment_gain: float = 12.0
    max_alignment_yaw: float = 10.0
    lost_grace_steps: int = 2
    reverse_after_steps: int = 36

    def __post_init__(self):
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("Invalid minimum confidence")
        if not 0 <= self.center_tolerance < 1:
            raise ValueError("Invalid center tolerance")
        if not all(
            math.isfinite(x) and x > 0
            for x in (
                self.scan_yaw,
                self.alignment_gain,
                self.max_alignment_yaw,
            )
        ):
            raise ValueError("Yaw parameters must be positive and finite")
        if type(self.lost_grace_steps) is not int or self.lost_grace_steps < 0:
            raise ValueError("Invalid loss grace period")
        if type(self.reverse_after_steps) is not int or self.reverse_after_steps < 1:
            raise ValueError("Invalid sweep length")


@dataclass(frozen=True)
class SearchMemory:
    has_seen_target: bool = False
    misses: int = 0
    direction: int = 1
    sweep_steps: int = 0


@dataclass(frozen=True)
class SearchCommand:
    mode: Literal["search", "align", "combat"]
    yaw_delta: float | None

    @property
    def use_combat_policy(self) -> bool:
        return self.mode == "combat"


class ClosedLoopSearchController:
    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()

    def step(
        self,
        detection: TargetDetection,
        memory: SearchMemory | None = None,
    ) -> tuple[SearchCommand, SearchMemory]:
        """Decide the next camera behavior from current visual feedback."""

        state = memory if memory is not None else SearchMemory()
        cfg = self.config

        visible = (
            detection.center_x is not None
            and detection.confidence >= cfg.min_confidence
        )

        if visible:
            x = detection.center_x
            assert x is not None

            direction = (
                state.direction
                if x == 0
                else (1 if x > 0 else -1)
            )

            state = SearchMemory(
                has_seen_target=True,
                misses=0,
                direction=direction,
                sweep_steps=0,
            )

            # Target centered: leave camera control to combat VPT.
            if abs(x) <= cfg.center_tolerance:
                return SearchCommand("combat", None), state

            # Target off-center: rotate toward its horizontal position.
            yaw = max(
                -cfg.max_alignment_yaw,
                min(cfg.max_alignment_yaw, cfg.alignment_gain * x),
            )
            return SearchCommand("align", yaw), state

        # A short missing detection should not immediately interrupt combat.
        misses = state.misses + 1

        if state.has_seen_target and misses <= cfg.lost_grace_steps:
            return (
                SearchCommand("combat", None),
                replace(state, misses=misses),
            )

        # Search with feedback on every frame, reversing after a full sweep.
        direction = state.direction
        sweep_steps = state.sweep_steps

        if sweep_steps >= cfg.reverse_after_steps:
            direction = -direction
            sweep_steps = 0

        return (
            SearchCommand("search", direction * cfg.scan_yaw),
            replace(
                state,
                misses=misses,
                direction=direction,
                sweep_steps=sweep_steps + 1,
            ),
        )
