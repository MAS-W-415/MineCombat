
"""Trainable high-level tactical Actor-Critic policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .contracts import RiskEstimate, TacticalDecision


@dataclass(frozen=True)
class HighLevelDecisionInfo:
    """Statistics of a high-level decision during rollout."""

    log_prob: float
    value: float


class HighLevelTacticalPolicy(nn.Module):
    """Learn a distribution over discrete tactical conditions.

    The policy consumes one feature vector per high-level
    decision. It does not produce native Minecraft actions.
    """

    def __init__(
        self,
        feature_dim: int,
        num_tactics: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if type(feature_dim) is not int or feature_dim < 1:
            raise ValueError("feature_dim must be positive")

        if type(num_tactics) is not int or num_tactics < 1:
            raise ValueError("num_tactics must be positive")

        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        self.feature_dim = feature_dim
        self.num_tactics = num_tactics

        # Shared feature extractor.
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor: tactical action distribution.
        self.actor_head = nn.Linear(
            hidden_dim,
            num_tactics,
        )

        # Critic: expected discounted high-level return.
        self.critic_head = nn.Linear(
            hidden_dim,
            1,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute tactical logits and high-level values.

        Args:
            features: [B, D]

        Returns:
            logits: [B, num_tactics]
            values: [B]
        """

        if features.ndim != 2:
            raise ValueError("features must have shape [B, D]")

        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}"
            )

        hidden = self.encoder(features)

        logits = self.actor_head(hidden)

        values = self.critic_head(hidden).squeeze(-1)

        return logits, values

    @torch.no_grad()
    def decide(
        self,
        features: Any,
        risk: RiskEstimate | None,
        memory: Any,
        *,
        deterministic: bool,
    ) -> tuple[TacticalDecision, HighLevelDecisionInfo]:
        """Choose a tactic using the learned Actor."""

        # The initial H2 model is feedforward.
        # Risk-aware decisions are reserved for H3.
        del risk, memory

        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )

        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected one feature vector of shape "
                f"({self.feature_dim},)"
            )

        # [D] -> [1, D]
        batch_features = features.unsqueeze(0)

        logits, values = self.forward(batch_features)

        distribution = Categorical(logits=logits[0])

        if deterministic:
            tactic = torch.argmax(logits[0])
        else:
            tactic = distribution.sample()

        tactic_id = int(tactic.item())

        probability = distribution.probs[tactic].item()

        decision = TacticalDecision(
            tactic_id=tactic_id,
            confidence=float(probability),
        )

        # Preserve the behavior-policy statistics for PPO.
        decision_info = HighLevelDecisionInfo(
            log_prob=float(
                distribution.log_prob(tactic).item()
            ),
            value=float(values[0].item()),
        )

        return decision, decision_info
