import re
from types import SimpleNamespace

import numpy as np

from minecombat.envs.callbacks.random_spawn import (
    RandomZombieSpawnCallback,
    ZombieSpawnConfig,
)


class FakeEnv:
    def __init__(self):
        self.commands = []

    def execute_cmd(self, command):
        self.commands.append(command)
        return {}, 0.0, False, {}

def make_sim(seed=42):
    return SimpleNamespace(
        np_random=np.random.default_rng(seed),
        env=FakeEnv(),
        _wrap_obs_info=lambda obs, info: (obs, info),
    )

def test_spawn_one_zombie():
    sim = make_sim()
    config = ZombieSpawnConfig(count=1)

    callback = RandomZombieSpawnCallback(config)
    callback.after_reset(sim, {}, {})

    assert len(sim.env.commands) == 2
    assert sim.env.commands[0].startswith("/tp @p")
    assert "summon minecraft:zombie" in sim.env.commands[1]

def test_spawn_is_reproducible():
    sim1 = make_sim(seed=42)
    sim2 = make_sim(seed=42)

    config = ZombieSpawnConfig(count=1)
    callback = RandomZombieSpawnCallback(config)

    callback.after_reset(sim1, {}, {})
    callback.after_reset(sim2, {}, {})
    print(sim1.env.commands)
    assert sim1.env.commands == sim2.env.commands


def test_no_ai_spawn_tag():
    sim = make_sim()
    config = ZombieSpawnConfig(count=1, no_ai=True)

    callback = RandomZombieSpawnCallback(config)
    callback.after_reset(sim, {}, {})

    assert "NoAI:1b" in sim.env.commands[1]
    assert "PersistenceRequired:1b" in sim.env.commands[1]


def test_fixed_front_spawn():
    sim = make_sim()
    config = ZombieSpawnConfig(
        count=1,
        min_distance=3.0,
        max_distance=4.0,
        spawn_angle_deg=90.0,
        player_yaw=0.0,
    )

    callback = RandomZombieSpawnCallback(config)
    callback.after_reset(sim, {}, {})

    assert sim.env.commands[0] == "/tp @p ~ ~ ~ 0.00 0.0"

    match = re.search(
        r"~(-?\d+\.\d+) ~2\.000 ~(-?\d+\.\d+)",
        sim.env.commands[1],
    )
    assert match is not None

    x = float(match.group(1))
    z = float(match.group(2))

    assert abs(x) < 1e-6
    assert 3.0 <= z <= 4.0