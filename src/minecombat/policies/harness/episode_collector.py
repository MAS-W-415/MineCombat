
"""Collect one complete episode for high-level PPO."""

from __future__ import annotations

from dataclasses import dataclass

from .high_level_session import HighLevelRolloutSession


@dataclass(frozen=True)
class EpisodeCollectionResult:
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    num_transitions: int


def collect_episode(
    env,
    session: HighLevelRolloutSession,
    *,
    max_steps: int = 600,
) -> EpisodeCollectionResult:
    """Collect one episode without updating policy weights."""

    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    if session.memory is not None:
        raise RuntimeError(
            "Previous episode has not finished"
        )

    before = len(session.collector.transitions)

    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(1, max_steps + 1):

        # High-level decision + low-level VPT action.
        action = session.act(obs)

        # Execute a real primitive action.
        next_obs, reward, terminated, truncated, _ = (
            env.step(action)
        )

        # Treat our own collection limit as truncation,
        # not as a genuine terminal state.
        reached_limit = (
            step == max_steps
            and not terminated
            and not truncated
        )

        terminated = bool(terminated)
        truncated = bool(truncated or reached_limit)

        # Store the primitive reward and episode boundary.
        session.observe(
            next_obs,
            reward,
            terminated=terminated,
            truncated=truncated,
        )

        total_reward += float(reward)

        if terminated or truncated:
            return EpisodeCollectionResult(
                steps=step,
                total_reward=total_reward,
                terminated=terminated,
                truncated=truncated,
                num_transitions=(
                    len(session.collector.transitions) - before
                ),
            )

        obs = next_obs

    raise RuntimeError("Unreachable episode state")
