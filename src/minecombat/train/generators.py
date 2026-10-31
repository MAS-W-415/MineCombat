
"""Pickle-safe generators for MineStudio online training."""

from __future__ import annotations


def env_generator(config_path: str):
    """Original flat PPO environment."""
    from minecombat.envs import make_zombie_env

    return make_zombie_env(config_path)


def policy_generator(repo_id: str):
    """Original flat VPT policy."""
    from minestudio.models import VPTPolicy
    from minestudio.utils import get_compute_device

    from .patches import patch_scaled_mse_head

    patch_scaled_mse_head()

    return VPTPolicy.from_pretrained(repo_id).to(
        get_compute_device()
    )


def conditioned_env_generator(
    config_path: str,
    num_tactics: int,
    tactic_id: int,
    sample_per_episode: bool,
):
    """Environment that records the tactic in each observation."""
    from minecombat.policies.harness.rollout_conditioning import (
        EpisodeTacticObservationWrapper,
    )

    env = env_generator(config_path)

    return EpisodeTacticObservationWrapper(
        env,
        num_tactics=num_tactics,
        tactic_id=tactic_id,
        sample_per_episode=sample_per_episode,
    )


def conditioned_policy_generator(
    repo_id: str,
    num_tactics: int,
):
    """Construct a trainable conditioned VPT."""
    from minestudio.models import VPTPolicy
    from minestudio.utils import get_compute_device

    from minecombat.policies.harness.conditioned_vpt import (
        ConditionedVPTPolicy,
    )

    from .patches import patch_scaled_mse_head

    patch_scaled_mse_head()

    base_policy = VPTPolicy.from_pretrained(repo_id)

    policy = ConditionedVPTPolicy(
        base_policy=base_policy,
        num_tactics=num_tactics,
    )

    return policy.to(get_compute_device())
