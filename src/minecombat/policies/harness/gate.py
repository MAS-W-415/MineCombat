"""Decision gates for hierarchical combat policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import RiskEstimate, TacticalDecision


@dataclass(frozen=True)
class FixedIntervalGate:
    """Refresh the tactical decision every N environment steps."""

    interval_steps: int = 20

    def __post_init__(self) -> None:
        if (
            type(self.interval_steps) is not int
            or self.interval_steps < 1
        ):
            raise ValueError(
                "interval_steps must be a positive integer"
            )

    def should_update(
        self,
        features: Any,
        risk: RiskEstimate | None,
        tactic: TacticalDecision,
        steps_since_decision: int,
        memory: Any,
    ) -> tuple[bool, Any]:
        """Return whether a new high-level decision is needed."""

        update = steps_since_decision >= self.interval_steps

        return update, memory