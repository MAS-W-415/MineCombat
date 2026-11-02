
"""Training loop for the high-level tactical policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .episode_collector import collect_episode
from .high_level_ppo import (
    HighLevelPPOConfig,
    update_high_level_ppo,
)
from .high_level_session import HighLevelRolloutSession
from .tactical_policy import HighLevelTacticalPolicy


@dataclass(frozen=True)
class HighLevelTrainConfig:
    num_updates: int = 2
    episodes_per_update: int = 2
    max_steps: int = 600

    def __post_init__(self):
        for value in (
            self.num_updates,
            self.episodes_per_update,
            self.max_steps,
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Training parameters must be positive integers"
                )


def train_high_level(
    env,
    session: HighLevelRolloutSession,
    optimizer: torch.optim.Optimizer,
    *,
    train_config: HighLevelTrainConfig,
    ppo_config: HighLevelPPOConfig,
    save_dir: str | Path,
) -> list[dict]:
    """Collect episodes, update high-level PPO, save checkpoints."""

    policy = session.runtime.tactical_policy

    if not isinstance(policy, HighLevelTacticalPolicy):
        raise TypeError(
            "Trainer requires HighLevelTacticalPolicy"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    total_env_steps = 0
    total_episodes = 0

    for update_idx in range(1, train_config.num_updates + 1):

        update_steps = 0
        update_rewards = []

        # 1. Collect on-policy data with fixed model weights.
        policy.eval()

        for _ in range(train_config.episodes_per_update):
            with torch.no_grad():
                result = collect_episode(
                    env,
                    session,
                    max_steps=train_config.max_steps,
                )

            update_steps += result.steps
            update_rewards.append(result.total_reward)
            total_episodes += 1

        total_env_steps += update_steps

        # 2. Update the high-level Actor-Critic.
        # The low-level VPT is not part of this optimizer.
        transitions = session.collector.transitions

        metrics = update_high_level_ppo(
            policy,
            transitions,
            optimizer,
            ppo_config,
        )

        # 3. The rollout is stale after PPO updates.
        session.clear()

        record = {
            "update": update_idx,
            "total_episodes": total_episodes,
            "total_env_steps": total_env_steps,
            "update_env_steps": update_steps,
            "mean_episode_reward": (
                sum(update_rewards) / len(update_rewards)
            ),
            **metrics,
        }

        history.append(record)

        # 4. Save high-level weights and optimizer state.
        torch.save(
            {
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update_idx,
                "total_env_steps": total_env_steps,
                "total_episodes": total_episodes,
                "train_config": asdict(train_config),
                "ppo_config": asdict(ppo_config),
            },
            save_dir / "high_level_latest.pt",
        )

        # Keep readable metrics alongside the checkpoint.
        (save_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print(
            f"[HH2] update={update_idx} "
            f"steps={total_env_steps} "
            f"transitions={metrics['num_transitions']} "
            f"loss={metrics['total_loss']:.4f}"
        )

    return history
