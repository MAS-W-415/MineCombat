
"""Evaluate a fine-tuned Minecraft Zombie detector."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ..train.common import resolve_path


def prepare_evaluation(
    checkpoint: str | Path,
    data: str | Path,
    output_dir: str | Path,
    *,
    imgsz: int = 320,
    batch: int = 8,
    device: str = "0",
) -> tuple[Path, dict[str, Any], Path]:
    """Validate paths and prepare Ultralytics val arguments."""

    if type(imgsz) is not int or imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a multiple of 32")

    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be positive")

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be non-empty")

    checkpoint = resolve_path(checkpoint)
    data_path = resolve_path(data)
    run_dir = resolve_path(output_dir)

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    if run_dir.exists():
        raise FileExistsError(
            f"Evaluation output already exists: {run_dir}"
        )

    config = yaml.safe_load(
        data_path.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml")

    if config.get("names") != {0: "zombie"}:
        raise ValueError("Expected class 0: zombie")

    root_value = config.get("path")

    if not isinstance(root_value, str):
        raise ValueError("Missing dataset root")

    root = Path(root_value).expanduser()

    if not root.is_absolute():
        root = data_path.parent / root

    root = root.resolve()

    relative = config.get("val")

    if not isinstance(relative, str):
        raise ValueError("Missing validation split")

    relative_path = Path(relative)

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Invalid validation path")

    val_dir = root / relative_path

    if not val_dir.is_dir() or not any(val_dir.glob("*.png")):
        raise ValueError("Validation split has no PNG images")

    options = {
        "data": str(data_path),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": False,
        "plots": True,
    }

    return checkpoint, options, run_dir


def evaluate_detector(
    checkpoint: str | Path,
    data: str | Path,
    output_dir: str | Path,
    *,
    imgsz: int = 320,
    batch: int = 8,
    device: str = "0",
    dry_run: bool = False,
    model_factory: Callable[[str], Any] | None = None,
) -> dict[str, float] | None:
    """Run YOLO validation and save scalar detection metrics."""

    checkpoint_path, options, run_dir = prepare_evaluation(
        checkpoint,
        data,
        output_dir,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    run_config = {
        "checkpoint": str(checkpoint_path),
        **options,
    }

    if dry_run:
        print(json.dumps(run_config, indent=2))
        return None

    if model_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install Ultralytics before real validation"
            ) from exc

        model_factory = YOLO

    model = model_factory(str(checkpoint_path))
    result = model.val(**options)

    metrics = {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }

    for name, value in metrics.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid validation metric: {name}")

    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Minecraft Zombie YOLO weights."
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    try:
        evaluate_detector(
            checkpoint=args.checkpoint,
            data=args.data,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
