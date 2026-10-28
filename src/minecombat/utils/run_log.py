"""Small reproducibility helpers for experiment runs."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass
class RunLog:
    """Create a timestamped run directory and save its metadata."""

    output_dir: str | Path
    name: str
    run_id: str = field(init=False)
    run_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{self.name}"
        self.run_dir = Path(self.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Any, filename: str = "config.yaml") -> Path:
        path = self.run_dir / filename
        if isinstance(config, (str, Path)):
            shutil.copyfile(config, path)
        elif hasattr(config, "to_yaml"):
            path.write_text(config.to_yaml(), encoding="utf-8")
        elif isinstance(config, dict):
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(str(config), encoding="utf-8")
        return path

    def save_metrics(self, metrics: dict[str, Any], filename: str = "metrics.json") -> Path:
        path = self.run_dir / filename
        path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        return path

    def save_records(self, records: Iterable[Any], filename: str = "episodes.csv") -> Path:
        from ..eval.metrics import to_csv

        return to_csv(records, self.run_dir / filename)

    def save_git_commit(self) -> Path | None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.run_dir,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        path = self.run_dir / "git_commit.txt"
        path.write_text(commit + "\n", encoding="utf-8")
        return path
