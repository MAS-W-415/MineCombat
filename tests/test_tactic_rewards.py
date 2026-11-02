
"""Tests for tactic-dependent auxiliary rewards."""

import pytest

from minecombat.policies.harness.tactic_rewards import (
    TacticalRewardCalculator,
    TacticalRewardConfig,
)


def test_aggressive_reward():
    calculator = TacticalRewardCalculator()

    calculator.reset({
        "health": 20.0,
        "damage_dealt": {"zombie": 100.0},
    })

    reward, bonus = calculator.compute(
        base_reward=1.0,
        info={
            "health": 18.0,
            "damage_dealt": {"zombie": 110.0},
        },
        tactic_id=0,
    )

    # 10 damage-stat points * 0.005 = 0.05
    assert bonus == pytest.approx(0.05)
    assert reward == pytest.approx(1.05)


def test_defensive_reward():
    calculator = TacticalRewardCalculator()

    calculator.reset({
        "health": 20.0,
        "damage_dealt": 100.0,
    })

    reward, bonus = calculator.compute(
        base_reward=1.0,
        info={
            "health": 18.0,
            "damage_dealt": 110.0,
        },
        tactic_id=1,
    )

    # 2 HP lost * -0.1 = -0.2
    assert bonus == pytest.approx(-0.2)
    assert reward == pytest.approx(0.8)


def test_reset_and_nested_damage_counter():
    calculator = TacticalRewardCalculator(
        TacticalRewardConfig(
            aggressive_damage_bonus=0.01,
        )
    )

    # Simulate cumulative counters that persist across episodes.
    calculator.reset({
        "health": 20.0,
        "custom": {"damage_dealt": 199.0},
    })

    reward, bonus = calculator.compute(
        base_reward=0.0,
        info={
            "health": 20.0,
            "custom": {"damage_dealt": 200.0},
        },
        tactic_id=0,
    )

    assert reward == pytest.approx(0.01)

    # Start a new episode with a later cumulative counter.
    calculator.reset({
        "health": 20.0,
        "custom": {"damage_dealt": 300.0},
    })

    reward, bonus = calculator.compute(
        base_reward=0.0,
        info={
            "health": 20.0,
            "custom": {"damage_dealt": 305.0},
        },
        tactic_id=0,
    )

    assert bonus == pytest.approx(0.05)
    assert reward == pytest.approx(0.05)
