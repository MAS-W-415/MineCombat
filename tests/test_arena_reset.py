
from types import SimpleNamespace

from minecombat.envs.callbacks.arena_reset import ArenaResetCallback


class FakeRng:
    def integers(self, low, high):
        return low


class FakeEnv:
    def __init__(self):
        self.commands = []

    def execute_cmd(self, command):
        self.commands.append(command)
        return {}, 0.0, False, {}


def make_sim(health):
    return SimpleNamespace(
        already_reset=True,
        info={"health": health},
        np_random=FakeRng(),
        env=FakeEnv(),
    )


def test_dead_player_requires_hard_reset():
    sim = make_sim(0)
    callback = ArenaResetCallback()

    result = callback.before_reset(sim, True)

    assert result is True
    assert sim.env.commands == []


def test_alive_player_uses_fast_reset():
    sim = make_sim(20)
    callback = ArenaResetCallback()

    result = callback.before_reset(sim, True)

    assert result is False
    assert "/kill @e[type=!player]" in sim.env.commands
    assert "/kill" not in sim.env.commands
