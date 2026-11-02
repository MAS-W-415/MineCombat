"""Tests for the shared combat statistics tracker."""

from __future__ import annotations

import pytest

from minecombat.envs.callbacks.combat_stats import CombatStatsTracker


def test_custom_damage_and_death_delta() -> None:
    tracker = CombatStatsTracker()
    tracker.reset(
        {
            "health": 20.0,
            "custom": {"damage_dealt": 100.0},
            "kill_entity": {"zombie": 0.0},
        }
    )

    step = tracker.update(
        {
            "health": 15.0,
            "custom": {"damage_dealt": 140.0},
            "kill_entity": {"zombie": 1.0},
        }
    )

    assert step.hp == pytest.approx(15.0)
    assert step.hp_lost == pytest.approx(5.0)
    assert step.damage_dealt == pytest.approx(40.0)
    assert step.kills == pytest.approx(1.0)
    assert step.death == pytest.approx(0.0)

    death = tracker.update(
        {
            "health": 0.0,
            "custom": {"damage_dealt": 140.0},
            "kill_entity": {"zombie": 1.0},
        }
    )

    assert death.hp_lost == pytest.approx(15.0)
    assert death.death == pytest.approx(1.0)
