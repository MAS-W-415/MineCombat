"""Fast reset for the zombie arena.

MineStudio's ``FastResetCallback`` uses ``/kill`` without a target, which kills
the player. That is acceptable for item-collection tasks but not for a combat
task where the player must start alive at full health. This callback keeps the
fast-reset idea while only killing non-player entities.
"""

from __future__ import annotations

try:
    from minestudio.simulator.callbacks.callback import MinecraftCallback
except ImportError:  # pragma: no cover
    class MinecraftCallback:  # type: ignore[no-redef]
        """Fallback base class used when MineStudio is not installed."""


class ArenaResetCallback(MinecraftCallback):
    """Reset the arena without killing the player.

    The first reset performs a normal hard reset. Later resets run a short
    command sequence, teleport the player to a random plains location, clear the
    inventory, and restore health.
    """

    def __init__(
        self,
        biome: str = "plains",
        random_tp_range: int = 200,
        time: str = "midnight",
        weather: str = "clear",
        do_mob_spawning: bool = False,
        do_daylight_cycle: bool = False,
    ) -> None:
        self.biome = biome
        self.random_tp_range = random_tp_range
        self.time = time
        self.weather = weather
        self.do_mob_spawning = do_mob_spawning
        self.do_daylight_cycle = do_daylight_cycle

    def _world_commands(self) -> list[str]:
        return [
            f"/time set {self.time}",
            f"/weather {self.weather}",
            f"/gamerule doMobSpawning {str(self.do_mob_spawning).lower()}",
            f"/gamerule doDaylightCycle {str(self.do_daylight_cycle).lower()}",
        ]

    def _player_commands(self) -> list[str]:
        return [
            "/clear @p",
            "/effect give @p minecraft:instant_health 1 10 true",
            "/effect give @p minecraft:saturation 1 10 true",
        ]

    def _random_location(self, sim) -> tuple[int, int]:
        half = max(1, self.random_tp_range // 2)
        rng = getattr(sim, "np_random", None)
        if rng is None:
            import numpy as np

            return int(np.random.randint(-half, half)), int(np.random.randint(-half, half))
        return int(rng.integers(-half, half)), int(rng.integers(-half, half))

    def _execute(self, sim, commands: list[str]):
        obs = info = None
        for command in commands:
            obs, _, _, info = sim.env.execute_cmd(command)
        return obs, info

    def before_reset(self, sim, reset_flag: bool) -> bool:
        if not getattr(sim, "already_reset", False):
            return reset_flag

        x, z = self._random_location(sim)
        commands = [
            "/kill @e[type=!player]",
            "/kill @e[type=item]",
            *self._world_commands(),
            *self._player_commands(),
            f"/teleportbiome @a {self.biome} {x} ~0 {z}",
        ]
        self._execute(sim, commands)
        return False

    def after_reset(self, sim, obs, info):
        commands = [*self._world_commands(), *self._player_commands()]
        cmd_obs, cmd_info = self._execute(sim, commands)
        if cmd_obs is not None and cmd_info is not None:
            obs, info = sim._wrap_obs_info(cmd_obs, cmd_info)
        return obs, info
