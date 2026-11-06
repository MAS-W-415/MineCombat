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


def test_entity_attributes_are_added_to_nbt():
    sim = make_sim()
    config = ZombieSpawnConfig(
        count=1,
        attributes={
            "generic.movement_speed": 0.15,
            "generic.attack_damage": 2.0,
        },
    )

    callback = RandomZombieSpawnCallback(config)
    callback.after_reset(sim, {}, {})

    command = sim.env.commands[1]
    assert (
        'Attributes:[{Name:"generic.movement_speed",Base:0.15d},'
        '{Name:"generic.attack_damage",Base:2.0d}]'
    ) in command


def test_spawn_angle_jitter_stays_in_range():
    sim = make_sim()

    config = ZombieSpawnConfig(
        count=20,
        min_distance=4.0,
        max_distance=4.0,
        spawn_angle_deg=90.0,
        spawn_angle_jitter_deg=30.0,
        player_yaw=0.0,
    )

    callback = RandomZombieSpawnCallback(config)
    _, info = callback.after_reset(sim, {}, {})

    angles = [spawn["angle"] for spawn in info["spawn_debug"]]
    assert len(angles) == 20
    for angle in angles:
        # 90 +/- 30 degrees, converted to radians.
        assert np.pi / 3.0 - 1e-9 <= angle <= 2.0 * np.pi / 3.0 + 1e-9
    assert len(set(angles)) > 1


def test_face_spawn_points_player_at_first_zombie():
    sim = make_sim()

    config = ZombieSpawnConfig(
        count=1,
        min_distance=4.0,
        max_distance=4.0,
        spawn_angle_deg=0.0,
        randomize_player_yaw=False,
        face_spawn=True,
    )

    callback = RandomZombieSpawnCallback(config)
    callback.after_reset(sim, {}, {})

    assert "/tp @p ~ ~ ~ -90.00 0.0" in sim.env.commands