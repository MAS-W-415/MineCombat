from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from minecombat.envs.callbacks.combat_reward import CombatRewardConfig
from minecombat.envs.callbacks.random_spawn import ZombieSpawnConfig
from minecombat.envs.zombie_arena import load_config

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_environment_config_is_consistent() -> None:
    cfg = load("configs/env/zombie_stage1.yaml")
    assert cfg["obs_size"] == [128, 128]
    assert cfg["action_type"] == "agent"
    assert cfg["zombie"]["count"] == 1
    assert cfg["zombie"]["min_distance"] < cfg["zombie"]["max_distance"]
    assert cfg["max_episode_steps"] > 0


def test_reward_section_matches_dataclass() -> None:
    cfg = load("configs/env/zombie_stage1.yaml")
    reward = CombatRewardConfig(**cfg["reward"])
    assert reward.target_mob == "zombie"
    assert reward.kill > 0


def test_zombie_section_matches_dataclass() -> None:
    cfg = load("configs/env/zombie_stage1.yaml")
    zombie = ZombieSpawnConfig(**cfg["zombie"])
    assert zombie.mob == "zombie"
    assert zombie.randomize_player_yaw is True


def test_ppo_config_has_single_gpu_profile() -> None:
    cfg = load("configs/ppo/zombie_vpt1x.yaml")
    assert cfg["rollout_config"]["num_rollout_workers"] == 1
    assert cfg["train_config"]["num_workers"] == 1
    assert cfg["rollout_config"]["num_gpus_per_worker"] + cfg["train_config"]["num_gpus_per_worker"] == pytest.approx(1.0)


def test_eval_config_paths_exist() -> None:
    cfg = load("configs/eval/zombie.yaml")
    assert (ROOT / cfg["env_config"]).exists()


def test_overlay_config_inherits_base() -> None:
    cfg = load_config("configs/env/zombie_stage1_easy.yaml")

    assert cfg["obs_size"] == [128, 128]
    assert cfg["zombie"]["min_distance"] == 3.0
    assert cfg["zombie"]["spawn_angle_deg"] == 90.0
    assert cfg["zombie"]["persistence_required"] is True
    assert cfg["reward"]["kill"] == 10.0


def test_overlay_reward_config() -> None:
    cfg = load_config(
        "configs/env/zombie_stage1_aggressive.yaml"
    )

    assert cfg["reward"]["kill"] == 20.0
    assert cfg["reward"]["damage_dealt"] == 0.02
    assert cfg["reward"]["target_mob"] == "zombie"
    assert cfg["zombie"]["count"] == 1


def test_overlay_can_extend_another_overlay() -> None:
    cfg = load_config(
        "configs/env/zombie_stage1_stationary_attack.yaml"
    )

    assert cfg["zombie"]["no_ai"] is True
    assert cfg["reward"]["attack"] == 0.02
    assert cfg["reward"]["kill"] == 10.0


def test_circular_base_raises(tmp_path) -> None:
    (tmp_path / "a.yaml").write_text(
        "base: b.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        "base: a.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")
