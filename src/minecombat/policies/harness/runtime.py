"""Runtime orchestration for hierarchical combat policies."""

from __future__ import annotations

from typing import Any

from .contracts import (
    ConditionedLowLevelPolicy,
    DecisionGate,
    FeatureAdapter,
    HarnessState,
    RiskEstimate,
    RiskPredictor,
    TacticalDecision,
    TacticalPolicy,
)


class HarnessRuntime:
    """Coordinate high-level tactics and low-level actions."""

    def __init__(
        self,
        tactical_policy: TacticalPolicy,
        low_policy: ConditionedLowLevelPolicy,
        gate: DecisionGate,
        feature_adapter: FeatureAdapter | None = None,
        risk_predictor: RiskPredictor | None = None,
    ) -> None:
        self.tactical_policy = tactical_policy
        self.low_policy = low_policy
        self.gate = gate
        self.feature_adapter = feature_adapter
        self.risk_predictor = risk_predictor

    def get_action(
        self,
        obs: Any,
        memory: HarnessState | None,
        *,
        deterministic: bool = True,
        input_shape: str = "*",
    ) -> tuple[Any, HarnessState]:
        """Execute one hierarchical policy step."""

        # 1. Restore episode-local memory.
        state = memory if memory is not None else HarnessState()

        if not isinstance(state, HarnessState):
            raise TypeError(
                "memory must be HarnessState or None"
            )

        # 2. Extract visual or temporal features.
        if self.feature_adapter is not None:
            features, feature_memory = (
                self.feature_adapter.extract(
                    obs,
                    state.feature_memory,
                )
            )
        else:
            features = obs
            feature_memory = state.feature_memory

        # 3. Estimate risk under the current tactic.
        risk = state.current_risk
        risk_memory = state.risk_memory

        if self.risk_predictor is not None:
            risk, risk_memory = self.risk_predictor.estimate(
                features,
                state.current_tactic,
                state.risk_memory,
            )

            if not isinstance(risk, RiskEstimate):
                raise TypeError(
                    "Risk predictor must return RiskEstimate"
                )

        # 4. Determine whether to refresh the tactic.
        should_update = state.current_tactic is None
        gate_memory = state.gate_memory

        if not should_update:
            should_update, gate_memory = (
                self.gate.should_update(
                    features,
                    risk,
                    state.current_tactic,
                    state.steps_since_decision,
                    state.gate_memory,
                )
            )

        # 5. Update or retain the tactical condition.
        if should_update:
            tactic, high_memory = self.tactical_policy.decide(
                features,
                risk,
                state.high_memory,
                deterministic=deterministic,
            )

            if not isinstance(tactic, TacticalDecision):
                raise TypeError(
                    "Tactical policy must return TacticalDecision"
                )

            steps_since_decision = 0

        else:
            tactic = state.current_tactic
            high_memory = state.high_memory
            steps_since_decision = state.steps_since_decision

        # 6. Low-level policy produces the native action.
        action, low_memory = self.low_policy.get_action(
            obs,
            state.low_memory,
            tactical_condition=tactic,
            deterministic=deterministic,
            input_shape=input_shape,
        )

        # 7. Construct the next episode-local state.
        next_state = HarnessState(
            low_memory=low_memory,
            high_memory=high_memory,
            feature_memory=feature_memory,
            risk_memory=risk_memory,
            gate_memory=gate_memory,
            current_tactic=tactic,
            current_risk=risk,
            steps_since_decision=steps_since_decision + 1,
            total_steps=state.total_steps + 1,
            last_decision_updated=should_update,
        )

        return action, next_state