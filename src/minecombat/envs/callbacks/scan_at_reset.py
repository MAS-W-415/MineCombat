"""Reset-time camera scan for search experiments."""

from __future__ import annotations

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class when MineStudio is not installed."""


class ScanAtResetCallback(MinecraftCallback):
    """Rotate the camera for a fixed number of steps after every reset.

    The button actions still come from the policy; only the camera delta is
    overridden. With the default 36 steps x 10 degrees this performs a full
    turn, giving the policy a chance to see an off-screen target.
    """

    def __init__(
        self,
        steps: int = 36,
        yaw_per_step: float = 10.0,
        pitch_per_step: float = 0.0,
    ) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        if yaw_per_step == 0.0:
            raise ValueError("yaw_per_step must be non-zero")

        self.steps = int(steps)
        self.yaw_per_step = float(yaw_per_step)
        self.pitch_per_step = float(pitch_per_step)
        self._remaining = 0

    def after_reset(self, sim, obs, info):
        self._remaining = self.steps
        return obs, info

    def before_step(self, sim, action):
        if self._remaining > 0:
            action["camera"] = (
                self.yaw_per_step,
                self.pitch_per_step,
            )
            self._remaining -= 1
        return action
