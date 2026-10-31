"""Interfaces and data structures for the hierarchical combat harness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TacticalDecision:
    """A high-level tactical condition, not a keyboard action."""

    tactic_id: int
    confidence: float | None = None

    def __post_init__(self):
        if type(self.tactic_id) is not int or self.tactic_id < 0:
            raise ValueError("tactic_id must be a non-negative integer")

        if self.confidence is not None:
            if (
                not math.isfinite(self.confidence)
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class RiskEstimate:
    """Predicted combat risk over a future horizon."""

    expected_hp_loss: float
    death_probability: float
    horizon_steps: int

    def __post_init__(self):
        if not math.isfinite(self.expected_hp_loss) or self.expected_hp_loss < 0:
            raise ValueError("expected_hp_loss must be non-negative")

        if (
            not math.isfinite(self.death_probability)
            or not 0.0 <= self.death_probability <= 1.0
        ):
            raise ValueError("death_probability must be between 0 and 1")

        if type(self.horizon_steps) is not int or self.horizon_steps < 1:
            raise ValueError("horizon_steps must be a positive integer")


@dataclass(frozen=True)
class HarnessState:
    """Episode-local memory for all hierarchical components."""

    low_memory: Any = None
    high_memory: Any = None
    feature_memory: Any = None
    risk_memory: Any = None
    gate_memory: Any = None

    current_tactic: TacticalDecision | None = None
    current_risk: RiskEstimate | None = None

    steps_since_decision: int = 0
    total_steps: int = 0
    last_decision_updated: bool = False


class TacticalPolicy(Protocol):
    """Interface of the high-level tactical policy."""

    def decide(
        self,
        features: Any,
        risk: RiskEstimate | None,
        memory: Any,
        *,
        deterministic: bool,
    ) -> tuple[TacticalDecision, Any]:
        ...


class ConditionedLowLevelPolicy(Protocol):
    """Interface of a VPT policy conditioned on a tactic."""

    def get_action(
        self,
        obs: Any,
        memory: Any,
        *,
        tactical_condition: TacticalDecision,
        deterministic: bool,
        input_shape: str,
    ) -> tuple[Any, Any]:
        ...


class DecisionGate(Protocol):
    """Interface for deciding when to refresh the tactic."""

    def should_update(
        self,
        features: Any,
        risk: RiskEstimate | None,
        tactic: TacticalDecision,
        steps_since_decision: int,
        memory: Any,
    ) -> tuple[bool, Any]:
        ...


class RiskPredictor(Protocol):
    """Interface for predicting future combat risk."""

    def estimate(
        self,
        features: Any,
        tactic: TacticalDecision | None,
        memory: Any,
    ) -> tuple[RiskEstimate, Any]:
        ...


class FeatureAdapter(Protocol):
    """Interface for extracting permitted visual features."""

    def extract(
        self,
        obs: Any,
        memory: Any,
    ) -> tuple[Any, Any]:
        ...