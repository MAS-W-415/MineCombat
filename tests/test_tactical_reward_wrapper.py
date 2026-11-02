
"""Integration tests for tactic-aware Gymnasium rewards."""

import gymnasium as gym
import numpy as np
import pytest

from minecombat.policies.harness.rollout_conditioning import (
    EpisodeTacticObservationWrapper,
)
from minecombat.policies.harness.tactical_reward_wrapper import (
    TacticalRewardWrapper,
)


class FakeCombatEnv(gym.Env):
    """One-step combat environment with cumulative damage stats."""

    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(16, 16, 3),
                dtype=np.uint8,
            )
        })

        self.action_space = gym.spaces.Discrete(2)
        self.episode = 0

    def _obs(self):
        return {
            "image": np.zeros((16, 16, 3), dtype=np.uint8)
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.episode += 1

        # The cumulative counter does not reset to zero.
        self.start_damage = self.episode * 100.0

        return self._obs(), {
            "health": 20.0,
            "custom": {
                "damage_dealt": self.start_damage,
            },
        }

    def step(self, action):
        info = {
            "health": 19.0,
            "custom": {
                "damage_dealt": self.start_damage + 10.0,
            },
        }

        return self._obs(), 1.0, True, False, info


def make_env(tactic_id):
    base_env = FakeCombatEnv()

    conditioned_env = EpisodeTacticObservationWrapper(
        base_env,
        num_tactics=2,
        tactic_id=tactic_id,
        sample_per_episode=False,
    )

    return TacticalRewardWrapper(conditioned_env)


@pytest.mark.parametrize(
    "tactic_id, expected_reward, expected_bonus",
    [
        (0, 1.05, 0.05),
        (1, 0.9, -0.1),
    ],
)
def test_tactical_wrapper_reward(
    tactic_id,
    expected_reward,
    expected_bonus,
):
    env = make_env(tactic_id)

    try:
        obs, info = env.reset()

        assert int(obs["tactic_ids"]) == tactic_id

        obs, reward, terminated, truncated, info = (
            env.step(0)
        )

        assert terminated
        assert not truncated

        assert reward == pytest.approx(expected_reward)
        assert info["reward_base"] == pytest.approx(1.0)

        assert info["reward_tactical_aux"] == pytest.approx(
            expected_bonus
        )

        assert info["reward_training"] == pytest.approx(
            expected_reward
        )

        assert info["tactic_id"] == tactic_id

    finally:
        env.close()


def test_wrapper_resets_reward_history():
    env = make_env(tactic_id=0)

    try:
        # First episode.
        env.reset()
        _, reward1, _, _, _ = env.step(0)

        # Second episode starts with a higher cumulative counter.
        env.reset()
        _, reward2, _, _, _ = env.step(0)

        assert reward1 == pytest.approx(1.05)
        assert reward2 == pytest.approx(1.05)

    finally:
        env.close()


from minecombat.train import generators


@pytest.mark.parametrize(
    "enable_rewards, expected_reward",
    [
        (False, 1.0),
        (True, 1.05),
    ],
)
def test_conditioned_generator_reward_toggle(
    monkeypatch,
    enable_rewards,
    expected_reward,
):
    """Verify that tactical rewards are opt-in."""

    # Replace real Minecraft creation with our fake environment.
    monkeypatch.setattr(
        generators,
        "env_generator",
        lambda config_path: FakeCombatEnv(),
    )

    env = generators.conditioned_env_generator(
        config_path="unused.yaml",
        num_tactics=2,
        tactic_id=0,
        sample_per_episode=False,
        enable_tactical_rewards=enable_rewards,
    )

    try:
        obs, info = env.reset()

        assert int(obs["tactic_ids"]) == 0

        obs, reward, terminated, truncated, info = (
            env.step(0)
        )

        assert terminated
        assert reward == pytest.approx(expected_reward)

        if enable_rewards:
            assert info["reward_tactical_aux"] == pytest.approx(
                0.05
            )
        else:
            assert "reward_tactical_aux" not in info

    finally:
        env.close()
