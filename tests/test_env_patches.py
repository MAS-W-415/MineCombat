"""Tests for the MineStudio observation compatibility patch."""

from __future__ import annotations

import pytest

pytest.importorskip("minestudio")

from minestudio.simulator.minerl.herobraine.hero.handlers.agent.observations.mc_base_stats import (  # noqa: E501
    ObserveFromFullStats,
)

from minecombat.envs.patches import patch_observe_from_full_stats  # noqa: E402


def test_custom_stat_is_read_from_custom_mapping():
    patch_observe_from_full_stats()
    handler = ObserveFromFullStats("damage_dealt")

    value = handler.from_hero({"custom": {"damage_dealt": 98}})

    assert value == pytest.approx(98.0)


def test_top_level_event_still_uses_original_lookup():
    patch_observe_from_full_stats()
    handler = ObserveFromFullStats("kill_entity")

    value = handler.from_hero({"kill_entity": {"zombie": 1}})

    assert value["zombie"] == pytest.approx(1.0)


def test_missing_custom_stat_is_empty():
    patch_observe_from_full_stats()
    handler = ObserveFromFullStats("damage_dealt")

    value = handler.from_hero({"custom": {}})

    assert value["anything"] == 0.0
