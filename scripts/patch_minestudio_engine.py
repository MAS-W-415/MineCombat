#!/usr/bin/env python3
"""Patch the MineStudio engine so left-click attack works immediately.

Root cause: ``net.minecraft.client.MouseHelper.grabMouse()`` sets
``Minecraft.leftClickCounter = 10000`` to suppress the click that focuses the
window. MineStudio never clears it, so the agent cannot attack for 500 seconds.

The script replaces the ``sipush 10000`` instruction in ``MouseHelper.class``
with ``iconst_0`` and updates the engine jar. It is idempotent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

JAR_REL = "engine/build/libs/mcprec-6.13.jar"
CLASS = "net/minecraft/client/MouseHelper.class"
PATTERN = b"\x11\x27\x10"  # sipush 10000
PATCHED = b"\x03\x00\x00"  # iconst_0; nop; nop


def mine_studio_dir() -> Path:
    try:
        from minestudio.utils import get_mine_studio_dir

        return Path(get_mine_studio_dir())
    except Exception:
        return Path("/tmp/MineStudio")


def ensure_engine(root: Path) -> Path:
    jar = root / JAR_REL
    if jar.exists():
        return jar

    print("Engine not found; downloading it from Hugging Face...")
    from minestudio.simulator.entry import check_engine

    check_engine(skip_confirmation=True)
    if not jar.exists():
        raise FileNotFoundError(f"Engine jar still missing: {jar}")
    return jar


def patch_jar(jar: Path) -> bool:
    with zipfile.ZipFile(jar, "r") as archive:
        class_bytes = archive.read(CLASS)

    count = class_bytes.count(PATTERN)
    if count == 0:
        print(f"Already patched: {jar}")
        return False
    if count != 1:
        raise RuntimeError(f"Expected one {PATTERN!r} in {CLASS}, found {count}")

    patched = class_bytes.replace(PATTERN, PATCHED)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        class_path = tmp_path / CLASS
        class_path.parent.mkdir(parents=True, exist_ok=True)
        class_path.write_bytes(patched)

        backup = jar.with_suffix(jar.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(jar, backup)

        subprocess.run(
            ["zip", "-u", str(jar), CLASS],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(f"Patched leftClickCounter in {jar}")
    return True


def main() -> int:
    root = mine_studio_dir()
    try:
        jar = ensure_engine(root)
        patch_jar(jar)
    except Exception as exc:
        print(f"Engine patch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
