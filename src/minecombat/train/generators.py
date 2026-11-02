
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
    enable_tactical_rewards: bool = False,
):
    """Build a conditioned PPO environment.

    Optionally apply tactic-specific auxiliary rewards.
    """

    from minecombat.policies.harness.rollout_conditioning import (
        EpisodeTacticObservationWrapper,
    )

    # Current auxiliary reward design supports two tactics.
    if enable_tactical_rewards and num_tactics != 2:
        raise ValueError(
            "Tactical rewards currently require num_tactics=2"
        )

    # Step 1: Create the original Minecraft environment.
    env = env_generator(config_path)

    # Step 2: Attach a tactic ID to observations.
    env = EpisodeTacticObservationWrapper(
        env,
        num_tactics=num_tactics,
        tactic_id=tactic_id,
        sample_per_episode=sample_per_episode,
    )

    # Step 3: Optionally modify rewards for low-level PPO.
    if enable_tactical_rewards:
        from minecombat.policies.harness.tactical_reward_wrapper import (
            TacticalRewardWrapper,
        )

        env = TacticalRewardWrapper(env)

    return env


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
