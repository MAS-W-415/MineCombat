"""Small helpers for reading MineStudio ``info`` dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def scalar(value: Any, default: float = 0.0) -> float:
    """Convert a scalar, numpy array, or one-element sequence to a float."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return float(np.asarray(value).reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def event_count(info: Mapping[str, Any], event: str, obj: str) -> float:
    """Read a cumulative event counter from MineStudio ``info``.

    MineStudio usually stores events as ``{"zombie": 2.0}``. Some simulator
    versions expose a plain scalar instead; in that case the scalar is used as
    a total count. The zombie arena contains only one mob type, so this fallback
    is safe for stage 1.
    """
    payload = info.get(event, 0.0)
    if isinstance(payload, Mapping):
        return scalar(payload.get(obj, 0.0))
    return scalar(payload)
