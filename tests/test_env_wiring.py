"""Tests for zombie environment callback wiring."""

from __future__ import annotations

import sys
from types import ModuleType

from minecombat.envs.callbacks import (
    ArenaResetCallback,
    CombatEpisodeCallback,
    CombatRewardCallback,
    RandomZombieSpawnCallback,
)
from minecombat.envs.zombie_arena import make_zombie_env


class FakeMinecraftSim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCommandsCallback:
    def __init__(self, commands):
        self.commands = commands


def install_fake_minestudio(monkeypatch):
    """Install lightweight substitutes for MineStudio imports."""

    minestudio = ModuleType("minestudio")
    simulator = ModuleType("minestudio.simulator")
    callbacks = ModuleType("minestudio.simulator.callbacks")

    minestudio.__path__ = []
    simulator.__path__ = []

    simulator.MinecraftSim = FakeMinecraftSim
    callbacks.CommandsCallback = FakeCommandsCallback

    monkeypatch.setitem(
        sys.modules,
        "minestudio",
        minestudio,
    )

    monkeypatch.setitem(
        sys.modules,
        "minestudio.simulator",
        simulator,
    )

    monkeypatch.setitem(
        sys.modules,
        "minestudio.simulator.callbacks",
        callbacks,
    )


def test_factory_registers_callbacks_in_order(monkeypatch):
    install_fake_minestudio(monkeypatch)

    env = make_zombie_env(
        "configs/env/zombie_stage1.yaml"
    )

    registered = env.kwargs["callbacks"]

    assert [type(cb) for cb in registered] == [
        ArenaResetCallback,
        FakeCommandsCallback,
        RandomZombieSpawnCallback,
        CombatRewardCallback,
        CombatEpisodeCallback,
    ]

    episode_callback = registered[4]

    assert episode_callback.target_mob == "zombie"
    assert episode_callback.required_kills == 1
    assert episode_callback.max_episode_steps == 600
    assert env.kwargs["action_type"] == "agent"


def test_factory_uses_zombie_count_for_success_goal(monkeypatch):
    install_fake_minestudio(monkeypatch)

    env = make_zombie_env(
        {
            "zombie": {
                "mob": "zombie",
                "count": 3,
            },
            "max_episode_steps": 200,
        }
    )

    registered = env.kwargs["callbacks"]

    spawn_callback = registered[2]
    episode_callback = registered[4]

    assert spawn_callback.config.count == 3
    assert episode_callback.required_kills == 3
    assert episode_callback.max_episode_steps == 200