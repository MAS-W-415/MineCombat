"""Evaluation metrics for combat episodes."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


@dataclass
class EpisodeRecord:
    seed: int
    win: bool
    death: bool
    timeout: bool
    steps: int
    hp_lost: float
    damage_dealt: float
    time_to_kill: int | None
    reward: float


def summarize(records: Iterable[EpisodeRecord]) -> dict[str, float | int | None]:
    records = list(records)
    if not records:
        raise ValueError("Cannot summarize an empty list of episodes")

    n = len(records)
    wins = [record for record in records if record.win]
    hp_losses = [record.hp_lost for record in records]
    damage_dealt = sum(record.damage_dealt for record in records)
    damage_taken = sum(hp_losses)

    summary: dict[str, float | int | None] = {
        "episodes": n,
        "win_rate": len(wins) / n,
        "death_rate": sum(record.death for record in records) / n,
        "timeout_rate": sum(record.timeout for record in records) / n,
        "stuck_rate": sum(record.timeout and record.damage_dealt <= 0 for record in records) / n,
        "mean_hp_lost": mean(hp_losses),
        "mean_hp_lost_on_win": mean([record.hp_lost for record in wins]) if wins else None,
        "mean_time_to_kill": mean([record.time_to_kill for record in wins]) if wins else None,
        "mean_steps": mean([record.steps for record in records]),
        "mean_reward": mean([record.reward for record in records]),
        "damage_ratio": damage_dealt / max(1e-6, damage_taken),
    }
    return summary


def to_csv(records: Iterable[EpisodeRecord], path: str | Path) -> Path:
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path
