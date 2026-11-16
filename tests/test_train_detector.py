
"""Tests for Zombie detector training entry point."""

import json
from pathlib import Path

import pytest
import yaml

from minecombat.data.train_detector import (
    prepare_training,
    train_detector,
)


def make_dataset(tmp_path):
    root = tmp_path / "dataset"

    for split in ("train", "val"):
        images = root / "images" / split
        images.mkdir(parents=True)
        (images / "sample.png").write_bytes(b"fake-png")

    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "zombie"},
    }

    data_path = root / "data.yaml"
    data_path.write_text(yaml.safe_dump(config))

    (root / "summary.json").write_text(
        json.dumps({
            "train_positive": 10,
            "val_positive": 2,
        })
    )

    return data_path


def test_training_preflight(tmp_path):
    data = make_dataset(tmp_path)

    options, run_dir = prepare_training(
        data,
        tmp_path / "runs" / "zombie_v1",
        epochs=5,
        imgsz=320,
        batch=4,
        device="cpu",
    )

    assert options["epochs"] == 5
    assert options["imgsz"] == 320
    assert options["batch"] == 4
    assert options["device"] == "cpu"
    assert options["data"] == str(data)
    assert run_dir.name == "zombie_v1"


def test_dry_run_does_not_load_model(tmp_path, capsys):
    data = make_dataset(tmp_path)
    output = tmp_path / "runs" / "dry_run"

    result = train_detector(
        data,
        output,
        dry_run=True,
        device="cpu",
    )

    config = json.loads(capsys.readouterr().out)

    assert result is None
    assert config["weights"] == "yolo11n.pt"
    assert config["data"] == str(data)
    assert not output.exists()


def test_training_with_fake_model(tmp_path):
    data = make_dataset(tmp_path)
    output = tmp_path / "runs" / "zombie_v1"

    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(("load", weights))

        def train(self, **kwargs):
            calls.append(("train", kwargs))

            run_dir = (
                Path(kwargs["project"]) / kwargs["name"]
            )
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)

            (weights_dir / "best.pt").write_bytes(
                b"fake-checkpoint"
            )

    best = train_detector(
        data,
        output,
        epochs=3,
        device="cpu",
        model_factory=FakeYOLO,
    )

    assert best == output / "weights" / "best.pt"
    assert best.is_file()
    assert calls[0] == ("load", "yolo11n.pt")
    assert calls[1][1]["epochs"] == 3

    run_config = json.loads(
        (output / "run_config.json").read_text()
    )

    assert run_config["best_checkpoint"] == str(best)


def test_rejects_existing_output(tmp_path):
    data = make_dataset(tmp_path)
    output = tmp_path / "runs" / "existing"
    output.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        prepare_training(data, output)


def test_rejects_wrong_class(tmp_path):
    data = make_dataset(tmp_path)

    config = yaml.safe_load(data.read_text())
    config["names"] = {0: "cow"}
    data.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="zombie"):
        prepare_training(data, tmp_path / "runs" / "test")


def test_rejects_invalid_imgsz(tmp_path):
    data = make_dataset(tmp_path)

    with pytest.raises(ValueError, match="imgsz"):
        prepare_training(
            data,
            tmp_path / "runs" / "invalid",
            imgsz=123,
        )
