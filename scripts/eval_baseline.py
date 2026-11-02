#!/usr/bin/env python3
"""Evaluate a flat policy on the zombie arena."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.eval.cli import main  # noqa: E402

if __name__ == "__main__":
    main(default_policy="random")
