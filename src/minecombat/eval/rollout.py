"""Rollout utilities for fixed-seed evaluation."""

from __future__ import annotations

import random

import numpy as np

from ..envs.callbacks._info import event_count, scalar
from .metrics import EpisodeRecord


def seed_env(env, seed: int) -> None:
    """Seed Python, NumPy, and the simulator before a reset.

    ``MinecraftSim.reset`` does not accept a seed argument, so evaluation seeds
    the global RNGs and the environment's ``np_random`` generator explicitly.
    """
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


def _select_action(policy, obs, memory, deterministic: bool):
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
    raise TypeError("Policy must be None, callable, or expose get_action()")


def evaluate(
    env,
    policy=None,
    episodes: int = 10,
    seed_start: int = 0,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> list[EpisodeRecord]:
    """Run fixed-seed episodes and return one record per episode."""
    records: list[EpisodeRecord] = []
    step_limit = max_steps if max_steps is not None else 10_000

    for episode in range(episodes):
        seed = seed_start + episode
        seed_env(env, seed)
        obs, info = env.reset()

        initial_hp = scalar(info.get("health", 20.0), default=20.0)
        initial_damage = event_count(info, "damage_dealt", "zombie")
        initial_kills = event_count(info, "kill_entity", "zombie")

        memory = None
        total_reward = 0.0
        steps = 0
        kill_step: int | None = None
        win = False
        death = False

        while steps < step_limit:
            action, memory = _select_action(policy, obs, memory, deterministic)
            if action is None:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            kills = event_count(info, "kill_entity", "zombie")
            if kills > initial_kills and kill_step is None:
                kill_step = steps
                win = True
            hp = scalar(info.get("health", initial_hp), default=initial_hp)
            if hp <= 0.0:
                death = True
            if terminated or truncated:
                break

        final_hp = scalar(info.get("health", initial_hp), default=initial_hp)
        final_damage = event_count(info, "damage_dealt", "zombie")

        records.append(
            EpisodeRecord(
                seed=seed,
                win=win,
                death=death,
                timeout=not win and not death,
                steps=steps,
                hp_lost=max(0.0, initial_hp - final_hp),
                damage_dealt=max(0.0, final_damage - initial_damage),
                time_to_kill=kill_step,
                reward=total_reward,
            )
        )

    return records
