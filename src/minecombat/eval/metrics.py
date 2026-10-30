"""Evaluation metrics for combat episodes."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


@dataclass
class EpisodeRecord:
    """Results collected from one combat episode."""

    seed: int
    win: bool
    death: bool
    timeout: bool
    steps: int
    hp_lost: float
    damage_dealt: float
    time_to_kill: int | None
    reward: float

    # Appended with a default for backward compatibility.
    outcome: str = "unknown"


def summarize(
    records: Iterable[EpisodeRecord],
) -> dict[str, float | int | None]:
    """Aggregate metrics across all evaluated episodes."""

    records = list(records)

    if not records:
        raise ValueError(
            "Cannot summarize an empty list of episodes"
        )

    n = len(records)

    wins = [record for record in records if record.win]

    hp_losses = [
        record.hp_lost for record in records
    ]

    kill_times = [
        record.time_to_kill
        for record in wins
        if record.time_to_kill is not None
    ]

    total_damage_dealt = sum(
        record.damage_dealt for record in records
    )

    total_damage_taken = sum(hp_losses)

    summary: dict[str, float | int | None] = {
        "episodes": n,

        "win_rate": len(wins) / n,

        "death_rate": sum(
            record.death for record in records
        ) / n,

        "timeout_rate": sum(
            record.timeout for record in records
        ) / n,

        "external_done_rate": sum(
            record.outcome == "external_done"
            for record in records
        ) / n,

        "eval_limit_rate": sum(
            record.outcome == "eval_limit"
            for record in records
        ) / n,

        "stuck_rate": sum(
            record.timeout and record.damage_dealt <= 0
            for record in records
        ) / n,

        # hp_lost now means cumulative observed HP loss.
        "mean_hp_lost": mean(hp_losses),

        "mean_damage_taken": mean(hp_losses),

        "mean_hp_lost_on_win": (
            mean([record.hp_lost for record in wins])
            if wins else None
        ),

        "mean_time_to_kill": (
            mean(kill_times) if kill_times else None
        ),

        "mean_steps": mean(
            record.steps for record in records
        ),

        "mean_reward": mean(
            record.reward for record in records
        ),

        "damage_ratio": (
            total_damage_dealt
            / max(1e-6, total_damage_taken)
        ),
    }

    return summary


def to_csv(
    records: Iterable[EpisodeRecord],
    path: str | Path,
) -> Path:
    """Write episode records to a CSV file."""

    records = list(records)
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not records:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(
        asdict(records[0]).keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))

    return path