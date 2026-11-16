
"""Tests for Zombie detector validation CLI."""

import json
from types import SimpleNamespace

import pytest
import yaml

from minecombat.data.eval_detector import (
    evaluate_detector,
    prepare_evaluation,
)


def make_inputs(tmp_path):
    dataset = tmp_path / "dataset"
    val_dir = dataset / "images" / "val"
    val_dir.mkdir(parents=True)

    (val_dir / "sample.png").write_bytes(b"fake-png")

    data = dataset / "data.yaml"
    data.write_text(
        yaml.safe_dump({
            "path": str(dataset),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "zombie"},
        })
    )

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-weights")

    return checkpoint, data


def test_preflight(tmp_path):
    checkpoint, data = make_inputs(tmp_path)
    output = tmp_path / "runs" / "eval_v1"

    ckpt, options, run_dir = prepare_evaluation(
        checkpoint,
        data,
        output,
        imgsz=320,
        batch=4,
        device="cpu",
    )

    assert ckpt == checkpoint
    assert options["split"] == "val"
    assert options["batch"] == 4
    assert run_dir == output


def test_dry_run(tmp_path, capsys):
    checkpoint, data = make_inputs(tmp_path)
    output = tmp_path / "runs" / "dry_run"

    result = evaluate_detector(
        checkpoint,
        data,
        output,
        device="cpu",
        dry_run=True,
    )

    config = json.loads(capsys.readouterr().out)

    assert result is None
    assert config["checkpoint"] == str(checkpoint)
    assert not output.exists()


def test_fake_yolo_validation(tmp_path):
    checkpoint, data = make_inputs(tmp_path)
    output = tmp_path / "runs" / "eval_v1"
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(("load", weights))

        def val(self, **kwargs):
            calls.append(("val", kwargs))
            return SimpleNamespace(
                box=SimpleNamespace(
                    mp=0.8,
                    mr=0.7,
                    map50=0.75,
                    map=0.5,
                )
            )

    result = evaluate_detector(
        checkpoint,
        data,
        output,
        device="cpu",
        model_factory=FakeYOLO,
    )

    assert result["precision"] == pytest.approx(0.8)
    assert result["recall"] == pytest.approx(0.7)

    assert calls[0] == ("load", str(checkpoint))
    assert calls[1][1]["split"] == "val"

    saved = json.loads(
        (output / "metrics.json").read_text()
    )
    assert saved == result
    assert (output / "run_config.json").is_file()


def test_rejects_missing_checkpoint(tmp_path):
    _, data = make_inputs(tmp_path)

    with pytest.raises(FileNotFoundError):
        prepare_evaluation(
            tmp_path / "missing.pt",
            data,
            tmp_path / "eval",
        )


def test_rejects_existing_output(tmp_path):
    checkpoint, data = make_inputs(tmp_path)
    output = tmp_path / "eval"
    output.mkdir()

    with pytest.raises(FileExistsError):
        prepare_evaluation(checkpoint, data, output)


def test_rejects_invalid_metrics(tmp_path):
    checkpoint, data = make_inputs(tmp_path)

    class BadYOLO:
        def __init__(self, weights):
            pass

        def val(self, **kwargs):
            return SimpleNamespace(
                box=SimpleNamespace(
                    mp=float("nan"),
                    mr=0.7,
                    map50=0.8,
                    map=0.6,
                )
            )

    with pytest.raises(ValueError, match="precision"):
        evaluate_detector(
            checkpoint,
            data,
            tmp_path / "bad_eval",
            model_factory=BadYOLO,
        )
