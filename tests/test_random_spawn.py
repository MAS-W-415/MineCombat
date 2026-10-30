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