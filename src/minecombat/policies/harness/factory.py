
"""Factory for building a conditioned VPT combat harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import RiskEstimate, TacticalDecision
from .gate import FixedIntervalGate
from .runtime import HarnessRuntime


class FixedTacticalPolicy:
    """Temporary high-level policy for H1 integration.

    It always selects the same tactical condition.
    This is NOT a learned tactical policy.
    """

    def __init__(self, tactic_id: int = 0) -> None:
        if type(tactic_id) is not int or tactic_id < 0:
            raise ValueError(
                "tactic_id must be a non-negative integer"
            )

        self.tactic_id = tactic_id

    def decide(
        self,
        features: Any,
        risk: RiskEstimate | None,
        memory: Any,
        *,
        deterministic: bool,
    ) -> tuple[TacticalDecision, Any]:
        decision = TacticalDecision(
            tactic_id=self.tactic_id,
        )

        return decision, memory


def build_conditioned_harness(
    repo_id: str = "CraftJarvis/MineStudio_VPT.foundation_model_1x",
    *,
    device: str | None = None,
    base_checkpoint: str | None = None,
    conditioned_checkpoint: str | None = None,
    num_tactics: int = 4,
    tactic_id: int = 0,
    interval_steps: int = 20,
) -> HarnessRuntime:
    """Build a Harness using a pretrained conditioned VPT.

    base_checkpoint:
        Optional flat VPT/PPO checkpoint loaded before
        constructing the conditioned model.

    conditioned_checkpoint:
        Optional checkpoint containing the complete
        ConditionedVPTPolicy state_dict.
    """

    if type(num_tactics) is not int or num_tactics < 1:
        raise ValueError("num_tactics must be positive")

    if type(tactic_id) is not int or not 0 <= tactic_id < num_tactics:
        raise ValueError("tactic_id is outside the tactic range")

    # Lazy imports keep the H0 runtime lightweight.
    import torch

    from minecombat.policies.vpt import load_vpt_policy

    from .conditioned_vpt import ConditionedVPTPolicy

    # 1. Load the pretrained backbone and action heads.
    base_policy = load_vpt_policy(
        repo_id=repo_id,
        device=device,
        checkpoint=base_checkpoint,
    )

    # 2. Reuse pretrained weights and attach the conditioner.
    low_policy = ConditionedVPTPolicy(
        base_policy=base_policy,
        num_tactics=num_tactics,
    )

    # 3. Optionally restore a previously trained
    #    fully conditioned policy.
    if conditioned_checkpoint is not None:
        checkpoint_path = Path(conditioned_checkpoint)

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        low_policy.load_state_dict(
            state,
            strict=True,
        )

    low_policy.eval()

    # 4. Temporary high-level baseline.
    high_policy = FixedTacticalPolicy(
        tactic_id=tactic_id,
    )

    # 5. Low-frequency tactical decision gate.
    gate = FixedIntervalGate(
        interval_steps=interval_steps,
    )

    # 6. Assemble the hierarchical runtime.
    return HarnessRuntime(
        tactical_policy=high_policy,
        low_policy=low_policy,
        gate=gate,
    )
