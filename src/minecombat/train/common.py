"""Shared helpers for training entry points."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[3]


def resolve_path(
    value: str | Path,
    root: Path | None = None,
) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root or project_root()) / path
    return path.resolve()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
