"""Learnable tactical conditioning for VPT hidden features."""

from __future__ import annotations

import torch
from torch import nn


class TacticalFeatureAdapter(nn.Module):
    """Inject tactical conditions into VPT hidden representations.

    Inputs:
        features: [B, T, D] floating-point tensor
        tactic_ids: [B, T] integer tensor (torch.long)

    Output:
        conditioned features: [B, T, D]
    """

    def __init__(
        self,
        num_tactics: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        if num_tactics < 1:
            raise ValueError("num_tactics must be positive")

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        self.num_tactics = num_tactics
        self.hidden_size = hidden_size

        # Convert discrete tactical IDs into learnable vectors.
        self.embedding = nn.Embedding(
            num_embeddings=num_tactics,
            embedding_dim=hidden_size,
        )

        # Project tactical embeddings into the VPT feature space.
        self.projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize without disturbing the pretrained VPT."""

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

        # Initially, the tactical residual is exactly zero.
        nn.init.zeros_(self.projection.weight)

    def forward(
        self,
        features: torch.Tensor,
        tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return features conditioned on tactical decisions."""

        if features.ndim != 3:
            raise ValueError(
                "features must have shape [B, T, D]"
            )

        batch_size, time_steps, hidden_size = features.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected feature dimension {self.hidden_size}, "
                f"got {hidden_size}"
            )

        if not features.is_floating_point():
            raise TypeError("features must be floating point")

        if tactic_ids.ndim != 2:
            raise ValueError(
                "tactic_ids must have shape [B, T]"
            )

        if tactic_ids.shape != (batch_size, time_steps):
            raise ValueError(
                "tactic_ids must match the batch and time "
                "dimensions of features"
            )

        if tactic_ids.dtype != torch.long:
            raise TypeError(
                "tactic_ids must have dtype torch.long"
            )

        # Keep the indices on the same device as the features.
        tactic_ids = tactic_ids.to(device=features.device)

        # [B, T] -> [B, T, D]
        tactical_embedding = self.embedding(tactic_ids)

        # [B, T, D] -> [B, T, D]
        tactical_residual = self.projection(tactical_embedding)

        # Preserve the input feature dtype under mixed precision.
        tactical_residual = tactical_residual.to(
            dtype=features.dtype
        )

        # Residual conditioning.
        conditioned_features = features + tactical_residual

        return conditioned_features