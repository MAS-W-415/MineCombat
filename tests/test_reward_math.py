from __future__ import annotations

import pytest

from minecombat.envs.callbacks.combat_reward import CombatRewardCalculator, CombatRewardConfig


def info(hp: float = 20.0, damage: float = 0.0, kills: float = 0.0):
    return {
        "health": hp,
        "damage_dealt": {"zombie": damage},
        "kill_entity": {"zombie": kills},
    }


def test_damage_dealt_reward() -> None:
    calc = CombatRewardCalculator()
    calc.reset(info())
    reward = calc.compute(info(damage=4.0))
    assert reward == pytest.approx(0.4 - 0.001)


def test_kill_reward() -> None:
    calc = CombatRewardCalculator()
    calc.reset(info())
    reward = calc.compute(info(kills=1.0))
    assert reward == pytest.approx(10.0 - 0.001)


def test_damage_taken_penalty() -> None:
    calc = CombatRewardCalculator()
    calc.reset(info(hp=20.0))
    reward = calc.compute(info(hp=16.0))
    assert reward == pytest.approx(-0.8 - 0.001)


def test_death_penalty_is_applied_once() -> None:
    calc = CombatRewardCalculator()
    calc.reset(info(hp=20.0))
    first = calc.compute(info(hp=0.0))
    second = calc.compute(info(hp=0.0))
    assert first == pytest.approx(-9.0 - 0.001)
    assert second == pytest.approx(-0.001)


def test_reward_clipping() -> None:
    calc = CombatRewardCalculator(CombatRewardConfig(kill=100.0, clip_max=20.0))
    calc.reset(info())
    assert calc.compute(info(kills=1.0)) == pytest.approx(20.0)

    calc = CombatRewardCalculator(CombatRewardConfig(death=-100.0, clip_min=-10.0))
    calc.reset(info())
    assert calc.compute(info(hp=0.0)) == pytest.approx(-10.0)


def test_scalar_event_fallback() -> None:
    calc = CombatRewardCalculator()
    calc.reset({"health": 20.0, "damage_dealt": 0.0})
    reward = calc.compute({"health": 20.0, "damage_dealt": 5.0})
    assert reward == pytest.approx(0.5 - 0.001)


def test_missing_events_are_neutral() -> None:
    calc = CombatRewardCalculator()
    calc.reset({"health": 20.0})
    assert calc.compute({"health": 20.0}) == pytest.approx(-0.001)


def test_other_mob_does_not_count() -> None:
    calc = CombatRewardCalculator()
    calc.reset(info())
    reward = calc.compute({"health": 20.0, "damage_dealt": {"skeleton": 5.0}})
    assert reward == pytest.approx(-0.001)
