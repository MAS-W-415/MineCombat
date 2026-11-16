
"""Train a Minecraft Zombie detector with Ultralytics YOLO."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ..train.common import resolve_path


def prepare_training(
    data: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 30,
    imgsz: int = 320,
    batch: int = 8,
    device: str = "0",
    workers: int = 2,
    seed: int = 42,
) -> tuple[dict[str, Any], Path]:
    """Validate dataset and return Ultralytics training arguments."""

    for name, value in (
        ("epochs", epochs),
        ("imgsz", imgsz),
        ("batch", batch),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be positive")

    if imgsz < 32 or imgsz % 32 != 0:
        raise ValueError("imgsz must be a multiple of 32")

    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be non-negative")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be non-negative")

    if not device.strip():
        raise ValueError("device cannot be empty")

    data_path = resolve_path(data)
    run_dir = resolve_path(output_dir)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    if run_dir.exists():
        raise FileExistsError(
            f"Training output already exists: {run_dir}"
        )

    config = yaml.safe_load(
        data_path.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError("Invalid YOLO data.yaml")

    if config.get("names") != {0: "zombie"}:
        raise ValueError(
            "Dataset must contain class 0: zombie"
        )

    root_value = config.get("path")
    if not isinstance(root_value, str):
        raise ValueError("Dataset path is missing")

    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = data_path.parent / root
    root = root.resolve()

    for split in ("train", "val"):
        relative = config.get(split)

        if not isinstance(relative, str):
            raise ValueError(f"Missing {split} split")

        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(f"Invalid {split} path")

        images_dir = root / relative_path

        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)

        if not any(images_dir.glob("*.png")):
            raise ValueError(f"No PNG images in {images_dir}")

    # Dataset Builder writes this summary after validation.
    summary_path = root / "summary.json"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    if (
        summary.get("train_positive", 0) < 1
        or summary.get("val_positive", 0) < 1
    ):
        raise ValueError(
            "Train and val both require positive zombie examples"
        )

    options = {
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "seed": seed,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": False,
        "plots": True,
        "save": True,
    }

    return options, run_dir


def train_detector(
    data: str | Path,
    output_dir: str | Path,
    *,
    weights: str = "yolo11n.pt",
    epochs: int = 30,
    imgsz: int = 320,
    batch: int = 8,
    device: str = "0",
    workers: int = 2,
    seed: int = 42,
    dry_run: bool = False,
    model_factory: Callable[[str], Any] | None = None,
) -> Path | None:
    """Train YOLO and return the best checkpoint path."""

    options, run_dir = prepare_training(
        data,
        output_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        seed=seed,
    )

    run_config = {
        "weights": weights,
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
                "Install Ultralytics: pip install ultralytics"
            ) from exc

        model_factory = YOLO

    model = model_factory(weights)

    model.train(**options)

    best = run_dir / "weights" / "best.pt"

    if not best.is_file():
        raise RuntimeError(
            f"Training finished but best.pt is missing: {best}"
        )

    run_config["best_checkpoint"] = str(best)

    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Zombie detector trained: {best}")

    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the MineCombat Zombie detector."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    try:
        train_detector(
            args.data,
            args.output_dir,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
