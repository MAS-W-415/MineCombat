
"""Tests for policy-frame data export."""

import json

import numpy as np
import pytest

from minecombat.data import policy_frames
from minecombat.data.policy_frames import PolicyFrameExporter


def make_obs():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 10, 20]
    return {"image": rgb}


def test_frame_stride_and_manifest(tmp_path, monkeypatch):
    captured = []

    def fake_save(path, rgb):
        captured.append((path, rgb.copy()))
        path.write_bytes(b"fake-png")

    monkeypatch.setattr(policy_frames, "_save_png", fake_save)

    exporter = PolicyFrameExporter(tmp_path, stride=10)

    for step in range(21):
        exporter.capture(make_obs(), seed=42, step=step)

    assert len(captured) == 3
    assert [r["step"] for r in exporter.records] == [0, 10, 20]
    assert captured[0][1][0, 0].tolist() == [255, 10, 20]

    manifest = exporter.write_manifest()
    records = [
        json.loads(line)
        for line in manifest.read_text().splitlines()
    ]

    assert len(records) == 3
    assert records[0]["width"] == 128
    assert records[0]["seed"] == 42
    assert (tmp_path / records[0]["image"]).is_file()


def test_invalid_rgb(tmp_path):
    exporter = PolicyFrameExporter(tmp_path)

    with pytest.raises(ValueError):
        exporter.capture({}, seed=0, step=0)

    with pytest.raises(ValueError):
        exporter.capture(
            {"image": np.zeros((10, 10), dtype=np.uint8)},
            seed=0,
            step=0,
        )


def test_invalid_stride(tmp_path):
    with pytest.raises(ValueError):
        PolicyFrameExporter(tmp_path, stride=0)


def test_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        policy_frames,
        "_save_png",
        lambda path, rgb: path.write_bytes(b"png"),
    )

    exporter = PolicyFrameExporter(tmp_path)
    exporter.capture(make_obs(), seed=1, step=0)

    with pytest.raises(FileExistsError):
        exporter.capture(make_obs(), seed=1, step=0)
