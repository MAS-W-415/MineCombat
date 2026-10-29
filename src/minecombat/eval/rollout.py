"""Rollout utilities for fixed-seed combat evaluation."""

from __future__ import annotations

import random

import numpy as np

from ..envs.callbacks._info import event_count, scalar
from .metrics import EpisodeRecord


FINAL_OUTCOMES = {
    "success",
    "death",
    "timeout",
    "external_done",
}


def seed_env(env, seed: int) -> None:
    """Seed the evaluation environment and random generators."""

    random.seed(seed)
    np.random.seed(seed)

    env.seed = seed
    env.np_random = np.random.default_rng(seed)

    base_env = getattr(env, "env", None)

    if base_env is not None and hasattr(base_env, "seed"):
        try:
            base_env.seed(seed)
        except Exception:
            pass


def _select_action(
    policy,
    obs,
    memory,
    deterministic: bool,
):
    """Obtain an action from a policy."""

    if policy is None:
        return None, None

    if hasattr(policy, "get_action"):
        action, memory = policy.get_action(
            obs,
            memory,
            deterministic=deterministic,
            input_shape="*",
        )
        return action, memory

    if callable(policy):
        return policy(obs), memory

    raise TypeError(
        "Policy must be None, callable, or expose get_action()"
    )


def _resolve_outcome(
    info: dict,
    *,
    final_hp: float,
    episode_kills: float,
    terminated: bool,
    truncated: bool,
) -> str:
    """Resolve the final episode outcome."""

    # Death takes priority over simultaneous kills.
    if final_hp <= 0.0:
        return "death"

    # Prefer the explicit outcome from CombatEpisodeCallback.
    reported_outcome = info.get("combat_outcome")

    if reported_outcome in FINAL_OUTCOMES:
        return reported_outcome

    # Compatibility fallback for older single-zombie environments.
    if episode_kills >= 1.0:
        return "success"

    if truncated:
        return "timeout"

    if terminated:
        return "external_done"

    # Evaluation's own step limit was reached.
    return "eval_limit"


def evaluate(
    env,
    policy=None,
    episodes: int = 10,
    seed_start: int = 0,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> list[EpisodeRecord]:
    """Evaluate a policy and collect one record per episode."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    step_limit = (
        max_steps if max_steps is not None else 10_000
    )

    records: list[EpisodeRecord] = []

    for episode in range(episodes):
        seed = seed_start + episode

        seed_env(env, seed)

        obs, info = env.reset()

        # Initial state snapshots.
        initial_hp = scalar(
            info.get("health", 20.0),
            default=20.0,
        )

        previous_hp = initial_hp

        previous_damage = event_count(
            info,
            "damage_dealt",
            "zombie",
        )

        initial_kills = event_count(
            info,
            "kill_entity",
            "zombie",
        )

        memory = None
        total_reward = 0.0
        total_damage_taken = 0.0
        total_damage_dealt = 0.0

        steps = 0
        terminated = False
        truncated = False

        # Run one complete episode.
        while steps < step_limit:
            action, memory = _select_action(
                policy,
                obs,
                memory,
                deterministic,
            )

            if action is None:
                action = env.action_space.sample()

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            steps += 1
            total_reward += float(reward)

            # Track cumulative observed HP loss.
            current_hp = scalar(
                info.get("health", previous_hp),
                default=previous_hp,
            )

            hp_loss = max(
                0.0,
                previous_hp - current_hp,
            )

            total_damage_taken += hp_loss
            previous_hp = current_hp

            # Track damage dealt using numeric snapshots.
            current_damage = event_count(
                info,
                "damage_dealt",
                "zombie",
            )

            new_damage = max(
                0.0,
                current_damage - previous_damage,
            )

            total_damage_dealt += new_damage
            previous_damage = current_damage

            # Respect both Gymnasium flags and task outcomes.
            if (
                terminated
                or truncated
                or info.get("combat_outcome") in FINAL_OUTCOMES
            ):
                break

        # Determine the final result.
        final_kills = event_count(
            info,
            "kill_entity",
            "zombie",
        )

        episode_kills = max(
            0.0,
            final_kills - initial_kills,
        )

        outcome = _resolve_outcome(
            info,
            final_hp=previous_hp,
            episode_kills=episode_kills,
            terminated=terminated,
            truncated=truncated,
        )

        win = outcome == "success"
        death = outcome == "death"
        timeout = outcome in {
            "timeout",
            "eval_limit",
        }

        # For a successful episode, the final step completes
        # the full target-kill objective.
        kill_step = steps if win else None

        record = EpisodeRecord(
            seed=seed,
            win=win,
            death=death,
            timeout=timeout,
            steps=steps,
            hp_lost=total_damage_taken,
            damage_dealt=total_damage_dealt,
            time_to_kill=kill_step,
            reward=total_reward,
            outcome=outcome,
        )

        records.append(record)

    return records