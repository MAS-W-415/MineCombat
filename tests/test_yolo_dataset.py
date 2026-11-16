
"""Tests for Minecraft YOLO dataset validation."""

import json

import pytest

from minecombat.data.yolo_dataset import (
    load_labeled_frames,
    read_yolo_labels,
    split_by_episode,
)


def add_sample(root, seed, step, label_text):
    frames = root / "policy_frames"
    labels = root / "labels"

    relative = f"seed_{seed:04d}/step_{step:05d}.png"

    image_path = frames / relative
    label_path = labels / relative.replace(".png", ".txt")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset validation checks file presence and label syntax.
    # PNG decoding is tested separately in the frame exporter.
    image_path.write_bytes(b"fake-png")
    label_path.write_text(label_text, encoding="utf-8")

    record = {
        "image": relative,
        "seed": seed,
        "step": step,
        "width": 128,
        "height": 128,
    }

    with (frames / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")

    return image_path, label_path


def test_load_positive_and_negative_samples(tmp_path):
    add_sample(
        tmp_path, seed=1, step=0,
        label_text="0 0.5 0.5 0.25 0.4\n",
    )
    add_sample(
        tmp_path, seed=2, step=0,
        label_text="",
    )

    samples = load_labeled_frames(
        tmp_path / "policy_frames",
        tmp_path / "labels",
    )

    assert len(samples) == 2
    assert len(samples[0].boxes) == 1
    assert samples[0].boxes[0].class_id == 0

    # Explicitly reviewed background frame.
    assert samples[1].boxes == ()


def test_missing_label_is_not_a_negative(tmp_path):
    _, label_path = add_sample(
        tmp_path, seed=1, step=0, label_text="",
    )
    label_path.unlink()

    with pytest.raises(FileNotFoundError, match="Missing annotation"):
        load_labeled_frames(
            tmp_path / "policy_frames",
            tmp_path / "labels",
        )


@pytest.mark.parametrize("invalid_label", [
    "1 0.5 0.5 0.2 0.2",
    "0 0.9 0.5 0.4 0.2",
    "0 0.5 0.5 0.0 0.2",
    "0 nan 0.5 0.2 0.2",
    "0 0.5 0.5 0.2",
])
def test_reject_invalid_yolo_boxes(tmp_path, invalid_label):
    label_path = tmp_path / "bad.txt"
    label_path.write_text(invalid_label)

    with pytest.raises(ValueError):
        read_yolo_labels(label_path)


def test_episode_split_has_no_leakage(tmp_path):
    for seed in range(6):
        for step in (0, 10):
            add_sample(
                tmp_path,
                seed=seed,
                step=step,
                label_text="0 0.5 0.5 0.2 0.3",
            )

    samples = load_labeled_frames(
        tmp_path / "policy_frames",
        tmp_path / "labels",
    )

    train, val = split_by_episode(
        samples,
        val_fraction=0.2,
        random_seed=42,
    )

    train_seeds = {sample.seed for sample in train}
    val_seeds = {sample.seed for sample in val}

    assert train_seeds.isdisjoint(val_seeds)
    assert len(train) + len(val) == 12
    assert len(val_seeds) == 2

    train2, val2 = split_by_episode(samples, random_seed=42)
    assert [s.seed for s in train] == [s.seed for s in train2]
    assert [s.seed for s in val] == [s.seed for s in val2]


def test_reject_unsafe_manifest_path(tmp_path):
    frames = tmp_path / "policy_frames"
    frames.mkdir()

    record = {
        "image": "../outside.png",
        "seed": 0,
        "step": 0,
        "width": 128,
        "height": 128,
    }

    (frames / "manifest.jsonl").write_text(
        json.dumps(record) + "\n"
    )

    with pytest.raises(ValueError, match="Invalid image path"):
        load_labeled_frames(frames, tmp_path / "labels")
