
"""Build an Ultralytics YOLO dataset from labeled policy frames."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from .yolo_dataset import (
    LabeledFrame,
    load_labeled_frames,
    split_by_episode,
)


def _copy_split(
    samples: list[LabeledFrame],
    root: Path,
    split: str,
) -> list[dict]:
    """Copy matching RGB images and annotations into one split."""

    records = []

    for sample in samples:
        # Flatten names while preserving episode and frame identity.
        stem = f"seed_{sample.seed:04d}_step_{sample.step:05d}"

        image_rel = Path("images") / split / f"{stem}.png"
        label_rel = Path("labels") / split / f"{stem}.txt"

        image_dst = root / image_rel
        label_dst = root / label_rel

        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(sample.image_path, image_dst)
        shutil.copy2(sample.label_path, label_dst)

        records.append({
            "split": split,
            "seed": sample.seed,
            "step": sample.step,
            "image": image_rel.as_posix(),
            "label": label_rel.as_posix(),
            "num_boxes": len(sample.boxes),
        })

    return records


def build_yolo_dataset(
    frames_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> dict[str, int]:
    """Validate annotations, split episodes, and build YOLO files."""

    root = Path(output_dir).expanduser().resolve()

    # Never silently overwrite a previous dataset.
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {root}"
        )

    # First validate all source files and human annotations.
    samples = load_labeled_frames(
        frames_dir,
        labels_dir,
    )

    train, val = split_by_episode(
        samples,
        val_fraction=val_fraction,
        random_seed=random_seed,
    )

    train_positive = sum(bool(s.boxes) for s in train)
    val_positive = sum(bool(s.boxes) for s in val)

    # A detection training experiment needs positive examples
    # in both splits.
    if train_positive == 0 or val_positive == 0:
        raise ValueError(
            "Both train and val must contain positive zombie frames. "
            "Collect more labeled episodes or change the split seed."
        )

    # Only create the output after source validation succeeds.
    records = []
    records.extend(_copy_split(train, root, "train"))
    records.extend(_copy_split(val, root, "val"))

    # Use an absolute root so training works from any directory
    # on the same machine.
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "zombie"},
    }

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    # Track which original episode every training image came from.
    (root / "split_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "train_images": len(train),
        "val_images": len(val),
        "train_positive": train_positive,
        "val_positive": val_positive,
        "train_episodes": len({s.seed for s in train}),
        "val_episodes": len({s.seed for s in val}),
    }

    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a labeled Minecraft Zombie YOLO dataset."
    )

    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    summary = build_yolo_dataset(
        frames_dir=args.frames_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        random_seed=args.seed,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
