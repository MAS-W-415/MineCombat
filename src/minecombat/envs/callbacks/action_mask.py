"""Action-mask callbacks for stage-1 combat."""

from __future__ import annotations

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class when MineStudio is not installed."""


class ActionMaskCallback(MinecraftCallback):
    """Mask actions that are not useful for stage-1 combat."""

    def __init__(self, mask_inventory: bool = True) -> None:
        self.mask_inventory = mask_inventory

    def before_step(self, sim, action):
        if self.mask_inventory and "inventory" in action:
            action["inventory"] = 0
        return action
