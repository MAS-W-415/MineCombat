
"""Gymnasium wrapper for tactic-dependent training rewards."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .tactic_rewards import (
    TacticalRewardCalculator,
    TacticalRewardConfig,
)


class TacticalRewardWrapper(gym.Wrapper):
    """Add tactic-dependent auxiliary rewards during PPO training."""

    def __init__(
        self,
        env: gym.Env,
        config: TacticalRewardConfig | None = None,
    ) -> None:
        super().__init__(env)

        if (
            not isinstance(env.observation_space, gym.spaces.Dict)
            or "tactic_ids" not in env.observation_space.spaces
        ):
            raise ValueError(
                "TacticalRewardWrapper requires tactic_ids "
                "in the observation space"
            )

        self.calculator = TacticalRewardCalculator(config)

    def reset(self, *args, **kwargs):
        """Reset the environment and reward history."""

        obs, info = self.env.reset(*args, **kwargs)

        # Initialize damage and HP snapshots for this episode.
        self.calculator.reset(info)

        return obs, info

    def step(self, action):
        """Apply auxiliary rewards to the environment reward."""

        obs, base_reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # Tactic is selected by the inner observation wrapper.
        tactic_value = np.asarray(obs["tactic_ids"])

        if (
            tactic_value.shape != ()
            or not np.issubdtype(tactic_value.dtype, np.integer)
        ):
            raise ValueError("Expected a scalar integer tactic_id")

        tactic_id = int(tactic_value.item())

        training_reward, auxiliary_reward = (
            self.calculator.compute(
                base_reward=base_reward,
                info=info,
                tactic_id=tactic_id,
            )
        )

        # Preserve original information and add training metrics.
        updated_info = dict(info)

        updated_info["reward_base"] = float(base_reward)
        updated_info["reward_tactical_aux"] = auxiliary_reward
        updated_info["reward_training"] = training_reward
        updated_info["tactic_id"] = tactic_id

        return (
            obs,
            training_reward,
            terminated,
            truncated,
            updated_info,
        )
