#!/usr/bin/env python3
"""Check and summarize saved evaluation runs without loading the training stack.

Only the Python standard library is used. Inputs are never modified; reports
are written to stdout. Matching recorded metadata is not proof that the
original environments or checkpoint contents were identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

REQUIRED_COLUMNS = {
    "seed", "win", "death", "timeout", "steps", "hp_lost",
    "damage_dealt", "time_to_kill", "reward",
}
IDENTITY_FIELDS = (
    "policy", "model", "checkpoint", "checkpoint_sha256", "model_revision",
    "tactic_id", "num_tactics", "deterministic", "max_steps",
    "config_sha256", "git_commit",
)
OUTCOMES = {"success", "death", "timeout", "eval_limit", "external_done", "unknown"}


class SummaryError(ValueError):
    """An input is invalid or the requested runs cannot be compared safely."""


def _number(value, name: str, *, integer: bool = False, nonnegative: bool = True):
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a number")
        result = int(value) if integer else float(value)
        if integer and not isinstance(value, str) and result != value:
            raise ValueError("not an integer")
        if not math.isfinite(result) or (nonnegative and result < 0):
            raise ValueError("out of range")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SummaryError(f"Invalid {name}: {value!r}") from exc
    return result


def _boolean(value: str | None, name: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise SummaryError(f"Invalid {name}: expected true/false or 1/0, got {value!r}")


def _read_episodes(path: Path) -> list[dict]:
    records = []
    seeds = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(headers)
        if missing:
            raise SummaryError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        if len(headers) != len(set(headers)):
            raise SummaryError(f"{path}: duplicate CSV column names")
        for line, row in enumerate(reader, start=2):
            location = f"{path}:{line}"
            if None in row:
                raise SummaryError(f"{location}: too many CSV fields")
            record = {
                key: _boolean(row[key], f"{location} {key}")
                for key in ("win", "death", "timeout")
            }
            if sum(record.values()) > 1:
                raise SummaryError(f"{location}: win/death/timeout are mutually exclusive")
            for key in ("seed", "steps"):
                record[key] = _number(row[key], f"{location} {key}", integer=True)
            for key in ("hp_lost", "damage_dealt", "reward"):
                record[key] = _number(
                    row[key], f"{location} {key}", nonnegative=key != "reward",
                )
            ttk = row["time_to_kill"]
            record["time_to_kill"] = (
                _number(ttk, f"{location} time_to_kill", integer=True)
                if ttk is not None and ttk.strip() else None
            )
            if record["time_to_kill"] is not None and (
                not record["win"] or record["time_to_kill"] > record["steps"]
            ):
                raise SummaryError(f"{location}: time_to_kill must belong to a win within steps")
            outcome = (row.get("outcome") or "unknown").strip()
            if not outcome:
                outcome = "unknown"
            if outcome not in OUTCOMES:
                raise SummaryError(f"{location}: unknown outcome {outcome!r}")
            expected_flag = {
                "success": "win", "death": "death",
                "timeout": "timeout", "eval_limit": "timeout",
            }.get(outcome)
            if expected_flag and not record[expected_flag]:
                raise SummaryError(f"{location}: outcome conflicts with {expected_flag}")
            if outcome == "external_done" and any(record[key] for key in ("win", "death", "timeout")):
                raise SummaryError(f"{location}: external_done conflicts with outcome flags")
            record["outcome"] = outcome
            if record["seed"] in seeds:
                raise SummaryError(f"{location}: duplicate seed {record['seed']} within a run")
            seeds.add(record["seed"])
            records.append(record)
    if not records:
        raise SummaryError(f"{path}: no episodes")
    return records


def _metrics(records: list[dict]) -> dict:
    """Independently cross-check saved metrics using the existing CSV meanings."""
    n = len(records)
    wins = [row for row in records if row["win"]]
    kill_times = [row["time_to_kill"] for row in wins if row["time_to_kill"] is not None]
    total_damage = sum(row["damage_dealt"] for row in records)
    total_hp = sum(row["hp_lost"] for row in records)
    result = {"episodes": n}
    for flag in ("win", "death", "timeout"):
        result[f"{flag}_rate"] = sum(row[flag] for row in records) / n
    for flag in ("death", "timeout"):
        result[f"zero_damage_{flag}_rate"] = sum(
            row[flag] and row["damage_dealt"] <= 0 for row in records
        ) / n
    result.update({
        "stuck_rate": result["zero_damage_timeout_rate"],
        "mean_hp_lost": total_hp / n,
        "mean_damage_taken": total_hp / n,
        "mean_hp_lost_on_win": mean(row["hp_lost"] for row in wins) if wins else None,
        "mean_time_to_kill": mean(kill_times) if kill_times else None,
        "mean_steps": mean(row["steps"] for row in records),
        "mean_reward": mean(row["reward"] for row in records),
        "damage_ratio": total_damage / max(1e-6, total_hp),
    })
    outcomes_known = all(row["outcome"] not in {"", "unknown"} for row in records)
    for outcome in ("external_done", "eval_limit"):
        result[f"{outcome}_rate"] = (
            sum(row["outcome"] == outcome for row in records) / n if outcomes_known else None
        )
    return result


def _reject_constant(value: str):
    raise SummaryError(f"Non-finite JSON number: {value}")


def _read_json(path: Path, warnings: list[str]) -> dict | None:
    if not path.exists():
        warnings.append(f"Missing {path.name}; corresponding metadata/check is unknown.")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except ValueError as exc:
        raise SummaryError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryError(f"{path}: expected a JSON object")
    return data


def _check_metrics(path: Path, saved: dict, computed: dict) -> None:
    for key in saved.keys() & computed.keys():
        actual, expected = saved[key], computed[key]
        # Old CSVs may not record outcome. Do not invent an outcome comparison.
        if expected is None and key in {"external_done_rate", "eval_limit_rate"}:
            continue
        if actual is None and expected is None:
            continue
        if actual is None or expected is None:
            raise SummaryError(f"{path}: {key} mismatch: saved={actual!r}, CSV={expected!r}")
        actual = _number(
            actual, f"{path} {key}", integer=key == "episodes", nonnegative=key != "mean_reward",
        )
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise SummaryError(f"{path}: {key} mismatch: saved={actual}, CSV={expected}")


def _load_run(path: Path) -> tuple[dict, set[int]]:
    warnings: list[str] = []
    records = _read_episodes(path / "episodes.csv")
    metrics = _metrics(records)
    for key, value in metrics.items():
        if value is not None:
            _number(value, f"{path} computed {key}", nonnegative=key != "mean_reward")
    saved = _read_json(path / "metrics.json", warnings)
    if saved is not None:
        _check_metrics(path / "metrics.json", saved, metrics)
        missing = {"episodes", "win_rate", "death_rate", "timeout_rate"} - saved.keys()
        if missing:
            warnings.append(f"metrics.json lacks checks for: {', '.join(sorted(missing))}.")
    policy = _read_json(path / "policy.json", warnings)
    metadata = {key: None for key in IDENTITY_FIELDS}
    known_fields = []
    for key in IDENTITY_FIELDS:
        if policy is not None and key in policy:
            value = policy[key]
            # An explicit null checkpoint means no checkpoint was supplied.
            # Null values for other fields provide no comparable identity.
            if value is None and key != "checkpoint":
                continue
            if key == "deterministic":
                if not isinstance(value, bool):
                    raise SummaryError(f"{path}: deterministic must be a boolean")
            elif key in {"tactic_id", "num_tactics", "max_steps"}:
                if type(value) is not int or value < (0 if key == "tactic_id" else 1):
                    raise SummaryError(f"{path}: invalid {key}: {value!r}")
            elif value is not None and (not isinstance(value, str) or not value.strip()):
                raise SummaryError(f"{path}: {key} must be a nonempty string")
            metadata[key] = value
            known_fields.append(key)
    if policy is not None and "episodes" in policy:
        if _number(policy["episodes"], "policy.json episodes", integer=True) != len(records):
            raise SummaryError(f"{path}: policy.json episode count disagrees with CSV")
    config_path = path / "config.yaml"
    if config_path.exists():
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if "config_sha256" in known_fields and metadata["config_sha256"] != config_hash:
            raise SummaryError(f"{path}: policy.json config_sha256 conflicts with config.yaml")
        metadata["config_sha256"] = config_hash
        known_fields.append("config_sha256")
    else:
        warnings.append("Missing config.yaml; any recorded config hash cannot be verified.")
    commit_path = path / "git_commit.txt"
    if commit_path.exists():
        commit = commit_path.read_text(encoding="utf-8").strip()
        if commit:
            if "git_commit" in known_fields and metadata["git_commit"] != commit:
                raise SummaryError(f"{path}: policy.json git_commit conflicts with git_commit.txt")
            metadata["git_commit"] = commit
            known_fields.append("git_commit")
    missing_metadata = sorted(set(IDENTITY_FIELDS) - set(known_fields))
    if missing_metadata:
        warnings.append(f"Unrecorded metadata (unknown): {', '.join(missing_metadata)}.")
    if metrics["external_done_rate"] is None:
        warnings.append("Some outcomes are unknown; outcome-specific rates cannot be checked.")
    run = {
        "run_id": path.name,
        "path": str(path),
        "episodes": len(records),
        "metrics": metrics,
        "metadata": metadata,
        "known_metadata_fields": sorted(set(known_fields)),
        "warnings": warnings,
    }
    return run, {row["seed"] for row in records}


def summarize_runs(run_dirs: list[Path], *, task: str) -> dict:
    """Summarize explicit repetitions of one task; never discover or mix runs."""
    if not task.strip():
        raise SummaryError("A nonempty task label is required")
    if not run_dirs:
        raise SummaryError("At least one run directory is required")
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(paths) != len(set(paths)):
        raise SummaryError("Duplicate run directory (including path aliases)")
    runs = []
    common_seeds = None
    identities = {}
    for path in paths:
        try:
            run, seeds = _load_run(path)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise SummaryError(f"{path}: {exc}") from exc
        if common_seeds is not None and seeds != common_seeds:
            raise SummaryError(f"{path}: seed set differs; these are not matched repetitions")
        common_seeds = seeds
        for key in run["known_metadata_fields"]:
            value = json.dumps(run["metadata"][key], sort_keys=True, allow_nan=False)
            if key in identities and identities[key] != value:
                raise SummaryError(f"{path}: conflicting {key}; summarize separate groups")
            identities[key] = value
        runs.append(run)
    aggregate_metrics = {}
    for key in runs[0]["metrics"]:
        if key == "episodes":
            continue
        values = [run["metrics"][key] for run in runs if run["metrics"][key] is not None]
        aggregate_metrics[key] = {
            "mean": mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "runs_with_value": len(values),
        }
    return {
        "schema_version": 1,
        "task": task.strip(),
        "runs": runs,
        "aggregate": {
            "runs": len(runs),
            "episodes": sum(run["episodes"] for run in runs),
            "unique_seeds": len(common_seeds),
            "per_run": aggregate_metrics,
        },
        "warnings": [
            "Task is a user-supplied label. Missing metadata does not establish equivalence.",
            "Config hashes compare saved bytes only; inherited YAML bases are not resolved.",
            "Checkpoint paths are recorded identities, not verification of weight contents.",
            "Means and ranges describe repeated runs, not confidence intervals or new unique seeds.",
            "Zero-damage failures do not establish target visibility or a search failure cause.",
        ],
    }


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def format_markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    lines = [
        f"# Evaluation summary: {_cell(report['task'])}", "",
        f"Runs: {aggregate['runs']}; episodes: {aggregate['episodes']}; "
        f"unique seeds: {aggregate['unique_seeds']} (reused across runs).", "",
        "| Run | Episodes | Win | Death | Timeout | Zero-damage death | Zero-damage timeout |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ("win_rate", "death_rate", "timeout_rate",
            "zero_damage_death_rate", "zero_damage_timeout_rate")
    for run in report["runs"]:
        rates = " | ".join(f"{run['metrics'][key]:.2%}" for key in keys)
        lines.append(f"| {_cell(run['run_id'])} | {run['episodes']} | {rates} |")
    lines.extend(["", "All failure proportions above use all episodes as the denominator.", "",
                  "| Metric | Mean across runs | Min | Max | Runs with a value |",
                  "|---|---:|---:|---:|---:|"])
    for key, stats in aggregate["per_run"].items():
        cells = []
        for name in ("mean", "min", "max"):
            value = stats[name]
            cells.append("unknown" if value is None else
                         f"{value:.2%}" if key.endswith("_rate") else f"{value:.6g}")
        lines.append(f"| {key} | {' | '.join(cells)} | {stats['runs_with_value']} |")
    lines.extend(["", "## Recorded identity and limitations", ""])
    for run in report["runs"]:
        lines.extend([f"### {_cell(run['run_id'])}", "", f"Directory: {_cell(run['path'])}", ""])
        for key in ("policy", "model", "checkpoint", "config_sha256", "git_commit"):
            value = (json.dumps(run["metadata"][key], ensure_ascii=False)
                     if key in run["known_metadata_fields"] else "unknown")
            lines.append(f"- {key}: {_cell(value)}")
        lines.extend(f"- Warning: {_cell(warning)}" for warning in run["warnings"])
        lines.append("")
    lines.extend(f"- {_cell(warning)}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Explicit run directories")
    parser.add_argument("--task", required=True, help="Label for this single comparison group")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)
    try:
        report = summarize_runs(args.run_dirs, task=args.task)
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(format_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
