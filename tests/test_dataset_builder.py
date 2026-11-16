
"""Tests for building a complete YOLO dataset."""

import json

import pytest
import yaml

from minecombat.data.dataset_builder import build_yolo_dataset


def add_sample(root, seed, step, positive=True):
    frames = root / "policy_frames"
    labels = root / "labels"

    relative = f"seed_{seed:04d}/step_{step:05d}"

    image = frames / f"{relative}.png"
    label = labels / f"{relative}.txt"

    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)

    # The builder only copies source images.
    # Real PNG encoding is tested by PolicyFrameExporter.
    image.write_bytes(b"fake-png")

    annotation = (
        "0 0.5 0.5 0.25 0.4\n"
        if positive
        else ""
    )
    label.write_text(annotation, encoding="utf-8")

    record = {
        "image": f"{relative}.png",
        "seed": seed,
        "step": step,
        "width": 128,
        "height": 128,
    }

    with (frames / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return image, label


def test_builds_yolo_directory_and_config(tmp_path):
    for seed in range(6):
        add_sample(tmp_path, seed, 0, positive=True)
        add_sample(tmp_path, seed, 10, positive=False)

    output = tmp_path / "yolo"

    summary = build_yolo_dataset(
        tmp_path / "policy_frames",
        tmp_path / "labels",
        output,
        val_fraction=0.33,
        random_seed=42,
    )

    assert summary["train_images"] == 8
    assert summary["val_images"] == 4
    assert summary["train_positive"] == 4
    assert summary["val_positive"] == 2

    config = yaml.safe_load(
        (output / "data.yaml").read_text()
    )

    assert config["path"] == str(output.resolve())
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["names"] == {0: "zombie"}

    records = json.loads(
        (output / "split_manifest.json").read_text()
    )
    assert len(records) == 12

    train_seeds = {
        r["seed"] for r in records if r["split"] == "train"
    }
    val_seeds = {
        r["seed"] for r in records if r["split"] == "val"
    }

    assert train_seeds.isdisjoint(val_seeds)

    for record in records:
        assert (output / record["image"]).is_file()
        assert (output / record["label"]).is_file()


def test_preserves_empty_negative_labels(tmp_path):
    for seed in (1, 2):
        add_sample(tmp_path, seed, 0, positive=True)
        add_sample(tmp_path, seed, 10, positive=False)

    output = tmp_path / "yolo"

    build_yolo_dataset(
        tmp_path / "policy_frames",
        tmp_path / "labels",
        output,
    )

    records = json.loads(
        (output / "split_manifest.json").read_text()
    )

    negatives = [r for r in records if r["num_boxes"] == 0]

    assert len(negatives) == 2
    for record in negatives:
        assert (output / record["label"]).read_text() == ""


def test_rejects_existing_output(tmp_path):
    output = tmp_path / "yolo"
    output.mkdir()
    (output / "keep.txt").write_text("important")

    with pytest.raises(FileExistsError):
        build_yolo_dataset(
            tmp_path / "policy_frames",
            tmp_path / "labels",
            output,
        )

    assert (output / "keep.txt").read_text() == "important"


def test_rejects_split_without_positive_samples(tmp_path):
    add_sample(tmp_path, 1, 0, positive=True)
    add_sample(tmp_path, 2, 0, positive=False)

    output = tmp_path / "yolo"

    with pytest.raises(ValueError, match="positive zombie"):
        build_yolo_dataset(
            tmp_path / "policy_frames",
            tmp_path / "labels",
            output,
        )

    assert not output.exists()
