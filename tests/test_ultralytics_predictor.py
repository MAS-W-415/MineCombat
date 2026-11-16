
"""Tests for the Ultralytics Zombie detection backend."""

from types import SimpleNamespace

import numpy as np
import pytest

from minecombat.policies.harness.target_detector import (
    RGBTargetDetector,
)
from minecombat.policies.harness.ultralytics_predictor import (
    UltralyticsBoxPredictor,
)


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class FakeModel:
    def __init__(self, names=None, boxes=None):
        self.names = names or {0: "zombie", 1: "cow"}
        self.boxes = boxes
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        return [
            SimpleNamespace(
                names=self.names,
                boxes=self.boxes,
            )
        ]


def make_boxes():
    return SimpleNamespace(
        xyxy=FakeTensor([
            [72, 10, 108, 70],
            [5, 10, 35, 70],
        ]),
        conf=FakeTensor([0.9, 0.99]),
        cls=FakeTensor([0, 1]),
    )


def test_yolo_predictor_converts_rgb_and_filters_classes():
    model = FakeModel(boxes=make_boxes())

    predictor = UltralyticsBoxPredictor(
        "fake.pt", model=model, device="cpu"
    )

    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 10, 20]

    boxes = predictor.predict(rgb)

    assert len(boxes) == 1
    assert boxes[0].label == "zombie"
    assert boxes[0].confidence == pytest.approx(0.9)

    # RGB -> BGR.
    assert model.last_kwargs["source"][0, 0].tolist() == [
        20, 10, 255
    ]

    assert model.last_kwargs["device"] == "cpu"


def test_works_with_existing_target_detector():
    predictor = UltralyticsBoxPredictor(
        "fake.pt",
        model=FakeModel(boxes=make_boxes()),
    )

    detector = RGBTargetDetector(predictor)

    detection = detector.detect({
        "image": np.zeros((80, 120, 3), dtype=np.uint8)
    })

    assert detection.center_x == pytest.approx(0.5)
    assert detection.confidence == pytest.approx(0.9)


def test_rejects_model_without_zombie_class():
    predictor = UltralyticsBoxPredictor(
        "fake.pt",
        model=FakeModel(names={0: "person"}, boxes=None),
    )

    with pytest.raises(ValueError, match="no zombie class"):
        predictor.predict(
            np.zeros((32, 32, 3), dtype=np.uint8)
        )


def test_no_detection():
    predictor = UltralyticsBoxPredictor(
        "fake.pt",
        model=FakeModel(boxes=None),
    )

    assert predictor.predict(
        np.zeros((32, 32, 3), dtype=np.uint8)
    ) == []


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        UltralyticsBoxPredictor(tmp_path / "missing.pt")
