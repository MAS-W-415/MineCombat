#!/usr/bin/env python3
"""Apply the local MineStudio launch-race workaround.

MineStudio waits for the Minecraft client to reach ``CLIENT enter state:
DORMANT`` but does not wait for the MalmoEnv TCP server to start listening. On
fast machines this causes ``Connection refused`` during reset. This script adds
a small grace period after the client is ready.

The patch is idempotent and only modifies the installed MineStudio package.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "MINECOMBAT_LAUNCH_GRACE"
NEEDLE = '            self._logger.info("Minecraft process ready")\n'


def main() -> int:
    try:
        from minestudio.simulator.minerl.env import malmo
    except ImportError:
        print("MineStudio is not installed; install it first.", file=sys.stderr)
        return 1

    path = Path(malmo.__file__)
    source = path.read_text(encoding="utf-8")

    if MARKER in source:
        print(f"Already patched: {path}")
        return 0

    if source.count(NEEDLE) != 1:
        print(
            f"Could not find the expected launch point in {path}. "
            "The MineStudio version may have changed; patch manually.",
            file=sys.stderr,
        )
        return 1

    replacement = (
        "            # MINECOMBAT_LAUNCH_GRACE: the MalmoEnv TCP server can\n"
        "            # start listening slightly after the client is ready.\n"
        "            import time as _time\n"
        "            _time.sleep(3.0)\n"
        "\n"
        '            self._logger.info("Minecraft process ready")\n'
    )
    path.write_text(source.replace(NEEDLE, replacement), encoding="utf-8")
    print(f"Patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
