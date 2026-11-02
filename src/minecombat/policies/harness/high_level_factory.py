
"""Construct the H2 trainable hierarchical combat harness."""

from __future__ import annotations

from pathlib import Path

from minecombat.policies.vpt import DEFAULT_MODEL

from .factory import build_conditioned_harness
from .gate import FixedIntervalGate
from .runtime import HarnessRuntime
from .tactical_policy import HighLevelTacticalPolicy
from .visual_features import VisualGridFeatureAdapter


def build_high_level_training_harness(
    conditioned_checkpoint: str | Path,
    *,
    repo_id: str = DEFAULT_MODEL,
    device: str | None = None,
    num_tactics: int = 2,
    interval_steps: int = 20,
    grid_size: int = 8,
    hidden_dim: int = 128,
) -> HarnessRuntime:
    """Build H2 with a frozen H1 conditioned VPT."""

    checkpoint = Path(conditioned_checkpoint).expanduser()

    # H2 must start from a trained H1 checkpoint.
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"H1 checkpoint not found: {checkpoint}"
        )

    if num_tactics != 2:
        raise ValueError(
            "Current H2 training requires num_tactics=2"
        )

    # 1. Reuse the existing H1 checkpoint loader.
    base_harness = build_conditioned_harness(
        repo_id=repo_id,
        device=device,
        conditioned_checkpoint=str(checkpoint),
        num_tactics=num_tactics,
        interval_steps=interval_steps,
    )

    low_policy = base_harness.low_policy

    # 2. Freeze the complete conditioned VPT.
    low_policy.requires_grad_(False)
    low_policy.eval()

    # 3. Create compact visual features for H2.
    feature_adapter = VisualGridFeatureAdapter(
        grid_size=grid_size,
    )

    # 4. Build a new trainable high-level Actor-Critic.
    high_policy = HighLevelTacticalPolicy(
        feature_dim=feature_adapter.feature_dim,
        num_tactics=num_tactics,
        hidden_dim=hidden_dim,
    ).to(low_policy.device)

    # 5. Build the complete hierarchical runtime.
    return HarnessRuntime(
        tactical_policy=high_policy,
        low_policy=low_policy,
        gate=FixedIntervalGate(
            interval_steps=interval_steps,
        ),
        feature_adapter=feature_adapter,
    )
