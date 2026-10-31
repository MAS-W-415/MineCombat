
"""A tactical-condition-aware MineStudio VPT policy.

H1: reuse pretrained VPT modules and inject a learnable
tactical residual before the action and value heads.
"""

from __future__ import annotations

import torch
from torch import nn
from minestudio.models import VPTPolicy

from .contracts import TacticalDecision
from .tactical_conditioning import TacticalFeatureAdapter


class ConditionedVPTPolicy(VPTPolicy):
    """VPT with high-level tactical conditioning."""

    def __init__(
        self,
        base_policy: VPTPolicy,
        num_tactics: int = 4,
    ) -> None:
        if not isinstance(base_policy, VPTPolicy):
            raise TypeError("base_policy must be a VPTPolicy")

        if type(num_tactics) is not int or num_tactics < 1:
            raise ValueError("num_tactics must be positive")

        # Reuse pretrained modules rather than construct
        # and randomly initialize another VPT backbone.
        #
        # This subclass deliberately initializes nn.Module
        # directly because it adopts an existing VPT model.
        nn.Module.__init__(self)

        self.net = base_policy.net
        self.pi_head = base_policy.pi_head
        self.value_head = base_policy.value_head

        # Required by VPTPolicy.initial_state().
        self.cached_init_states = {}

        hidden_size = self.net.output_latent_size()

        self.conditioner = TacticalFeatureAdapter(
            num_tactics=num_tactics,
            hidden_size=hidden_size,
        )

        # Match the pretrained model's device.
        self.conditioner.to(self.device)
        self.train(base_policy.training)

    def _resolve_tactic_ids(
        self,
        input: dict,
        tactical_condition: TacticalDecision | None,
    ) -> torch.Tensor:
        """Resolve the tactical condition for a [B, T] batch."""

        batch_size, time_steps = input["image"].shape[:2]

        if tactical_condition is not None:
            if "tactic_ids" in input:
                raise ValueError(
                    "Provide tactical_condition or tactic_ids, not both"
                )

            if not isinstance(tactical_condition, TacticalDecision):
                raise TypeError(
                    "tactical_condition must be TacticalDecision"
                )

            tactic_id = tactical_condition.tactic_id

            if tactic_id >= self.conditioner.num_tactics:
                raise ValueError(
                    f"tactic_id {tactic_id} is out of range"
                )

            return torch.full(
                (batch_size, time_steps),
                tactic_id,
                dtype=torch.long,
                device=self.device,
            )

        # Future PPO training can provide a different tactic
        # for each timestep in a batch.
        if "tactic_ids" not in input:
            raise ValueError(
                "Missing tactical_condition or input['tactic_ids']"
            )

        tactic_ids = input["tactic_ids"]

        if not isinstance(tactic_ids, torch.Tensor):
            raise TypeError("tactic_ids must be a tensor")

        return tactic_ids.to(device=self.device)

    
    def forward(
        self,
        input: dict,
        state_in=None,
        *,
        tactical_condition: TacticalDecision | None = None,
        context: dict | None = None,
        **kwargs,
    ):
        """Run conditioned VPT for inference or PPO training."""

        batch_size, time_steps = input["image"].shape[:2]

        # Recover the condition used when collecting the action.
        tactic_ids = self._resolve_tactic_ids(
            input,
            tactical_condition,
        )

        if state_in is None:
            state_in = self.initial_state(batch_size)

        # During PPO training, context["first"] identifies
        # episode boundaries inside a trajectory.
        if context is None:
            first = torch.zeros(
                (batch_size, time_steps),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            first = context.get("first")

            if not isinstance(first, torch.Tensor):
                raise TypeError(
                    "context['first'] must be a tensor"
                )

            if first.shape != (batch_size, time_steps):
                raise ValueError(
                    "context['first'] must have shape [B, T]"
                )

            first = first.to(
                device=self.device,
                dtype=torch.bool,
            )

        # Pretrained visual and recurrent backbone.
        (pi_h, value_h), state_out = self.net(
            input,
            state_in,
            context={"first": first},
        )

        # Trainable tactical conditioning.
        conditioned_pi_h = self.conditioner(
            pi_h,
            tactic_ids,
        )

        conditioned_value_h = self.conditioner(
            value_h,
            tactic_ids,
        )

        # Reuse pretrained action and value heads.
        return {
            "pi_logits": self.pi_head(conditioned_pi_h),
            "vpred": self.value_head(conditioned_value_h),
        }, state_out
