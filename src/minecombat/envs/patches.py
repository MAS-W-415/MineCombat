"""Compatibility patches for MineStudio observations.

MineStudio 1.1.6's ``ObserveFromFullStats`` looks up a stat at the top level of
the info dict. The engine emits ``custom`` stats (including ``damage_dealt``)
under ``info["custom"]``, so the damage stat was silently dropped and the dense
damage reward stayed at zero.

The patch only changes the lookup for stats that are absent at the top level
but present under ``custom``. ``kill_entity`` and other top-level stats keep the
original behavior.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np


def patch_observe_from_full_stats() -> None:
    """Read ``custom`` stats from ``info["custom"]``."""

    try:
        from minestudio.simulator.minerl.herobraine.hero.handlers.agent.observations.mc_base_stats import (  # noqa: E501
            ObserveFromFullStats,
        )
    except ImportError:
        return

    if getattr(ObserveFromFullStats, "_minecombat_patched", False):
        return

    original_from_hero = ObserveFromFullStats.from_hero

    def from_hero(self, info):
        if self.stat_key in info:
            return original_from_hero(self, info)

        custom = info.get("custom")
        if isinstance(custom, Mapping) and self.stat_key in custom:
            return float(custom[self.stat_key])

        return defaultdict(lambda: np.zeros((), dtype=float))

    ObserveFromFullStats.from_hero = from_hero
    ObserveFromFullStats._minecombat_patched = True
