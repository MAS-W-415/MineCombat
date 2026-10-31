
"""Training-time tactical observation conditioning.

Attach the tactic used for action selection to the observation,
so rollout and PPO training consume the same tactical ID.

This is an H1 data-flow baseline, not a learned tactical policy.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class EpisodeTacticObservationWrapper(gym.Wrapper):
    """Add an episode-level tactical condition to observations."""

    def __init__(
        self,
        env: gym.Env,
        num_tactics: int = 4,
        tactic_id: int = 0,
        sample_per_episode: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(env)

        if type(num_tactics) is not int or num_tactics < 1:
            raise ValueError("num_tactics must be a positive integer")

        if type(tactic_id) is not int or not 0 <= tactic_id < num_tactics:
            raise ValueError("tactic_id is outside the tactic range")

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("Expected a Dict observation space")

        if "tactic_ids" in env.observation_space.spaces:
            raise ValueError("Observation already contains tactic_ids")

        self.num_tactics = num_tactics
        self.tactic_id = tactic_id
        self.sample_per_episode = sample_per_episode

        self.rng = np.random.default_rng(seed)
        self.current_tactic: int | None = None

        # Extend the observation specification.
        spaces = dict(env.observation_space.spaces)

        spaces["tactic_ids"] = gym.spaces.Discrete(num_tactics)

        self.observation_space = gym.spaces.Dict(spaces)

    def _select_episode_tactic(self) -> int:
        """Choose a training condition for the new episode."""

        if self.sample_per_episode:
            return int(self.rng.integers(self.num_tactics))

        return self.tactic_id

    def _tag_observation(self, obs: dict) -> dict:
        """Attach the current tactic without modifying source obs."""

        if self.current_tactic is None:
            raise RuntimeError("Call reset() before requesting observations")

        tagged = dict(obs)

        # Scalar int64 becomes [B, T] after MineStudio stacking.
        tagged["tactic_ids"] = np.asarray(
            self.current_tactic,
            dtype=np.int64,
        )

        return tagged

    def reset(self, *args, **kwargs):
        """Reset the environment and choose the episode's tactic."""

        obs, info = self.env.reset(*args, **kwargs)

        self.current_tactic = self._select_episode_tactic()

        return self._tag_observation(obs), info

    def step(self, action):
        """Keep the same tactic throughout the episode."""

        obs, reward, terminated, truncated, info = self.env.step(action)

        return (
            self._tag_observation(obs),
            reward,
            terminated,
            truncated,
            info,
        )
