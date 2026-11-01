
"""Collect high-level training data from HarnessRuntime."""

from __future__ import annotations

from .runtime import HarnessRuntime
from .tactical_policy import HighLevelDecisionInfo
from .tactical_rollout import TacticalRolloutCollector


class HighLevelRolloutSession:
    """Connect Harness actions and environment rewards.

    Use one session per environment.
    Each act() must be followed by exactly one observe().
    """

    def __init__(self, runtime: HarnessRuntime):
        if runtime.feature_adapter is None:
            raise ValueError(
                "High-level training requires a feature adapter"
            )

        self.runtime = runtime
        self.collector = TacticalRolloutCollector()

        self.memory = None
        self._awaiting_reward = False

    def act(self, obs, *, input_shape: str = "*"):
        """Sample an action and record high-level decisions."""

        if self._awaiting_reward:
            raise RuntimeError(
                "Call observe() before the next act()"
            )

        action, next_memory = self.runtime.get_action(
            obs,
            self.memory,
            deterministic=False,
            input_shape=input_shape,
        )

        if next_memory.last_decision_updated:
            info = next_memory.high_memory

            if not isinstance(info, HighLevelDecisionInfo):
                raise TypeError(
                    "High-level policy must provide "
                    "HighLevelDecisionInfo"
                )

            log_prob = info.log_prob
            value = info.value

        else:
            log_prob = None
            value = None

        self.collector.on_action(
            state=next_memory,
            features=next_memory.last_features,
            behavior_log_prob=log_prob,
            behavior_value=value,
        )

        self.memory = next_memory
        self._awaiting_reward = True

        return action

    def observe(
        self,
        next_obs,
        reward: float,
        *,
        terminated: bool = False,
        truncated: bool = False,
    ):
        """Record the reward for the action just executed."""

        if not self._awaiting_reward:
            raise RuntimeError(
                "Call act() before observe()"
            )

        done = terminated or truncated

        if done:
            # Final state feature is needed for bootstrap.
            next_features, _ = (
                self.runtime.feature_adapter.extract(
                    next_obs,
                    self.memory.feature_memory,
                )
            )

        self.collector.record_reward(reward)

        if done:
            self.collector.finish_episode(
                next_features=next_features,
                terminated=terminated,
                truncated=truncated,
            )

            # Do not carry recurrent memory into a new episode.
            self.memory = None

        self._awaiting_reward = False

    def clear(self):
        """Discard completed trajectories between rollouts."""

        if self._awaiting_reward or self.memory is not None:
            raise RuntimeError(
                "Finish the episode before clearing"
            )

        self.collector.clear()
