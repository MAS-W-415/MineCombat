
import numpy as np
import pytest

from minecombat.policies.harness.target_detector import (
    DetectionBox,
    RGBTargetDetector,
)


class FakePredictor:
    def __init__(self, boxes):
        self.boxes = boxes
        self.last_image = None

    def predict(self, rgb_image):
        self.last_image = rgb_image
        return self.boxes


def make_obs():
    return {
        "image": np.zeros((80, 120, 3), dtype=np.uint8)
    }


def test_detects_highest_confidence_zombie():
    predictor = FakePredictor([
        DetectionBox(5, 10, 35, 70, 0.7),
        DetectionBox(72, 10, 108, 70, 0.9),
    ])

    detector = RGBTargetDetector(predictor)
    detection = detector.detect(make_obs())

    assert detection.center_x == pytest.approx(0.5)
    assert detection.confidence == pytest.approx(0.9)
    assert predictor.last_image.shape == (80, 120, 3)


def test_ignores_other_classes_and_low_confidence():
    predictor = FakePredictor([
        DetectionBox(10, 10, 40, 60, 0.99, "cow"),
        DetectionBox(70, 10, 100, 60, 0.2, "zombie"),
    ])

    detection = RGBTargetDetector(predictor).detect(make_obs())

    assert detection.center_x is None
    assert detection.confidence == 0.0


def test_ignores_tiny_and_outside_boxes():
    predictor = FakePredictor([
        DetectionBox(-100, 0, -10, 40, 0.99),
        DetectionBox(10, 10, 11, 11, 0.95),
    ])

    detection = RGBTargetDetector(predictor).detect(make_obs())

    assert detection.center_x is None


def test_empty_prediction():
    detector = RGBTargetDetector(FakePredictor([]))

    assert detector.detect(make_obs()).center_x is None


def test_invalid_inputs():
    detector = RGBTargetDetector(FakePredictor([]))

    with pytest.raises(ValueError):
        detector.detect({})

    with pytest.raises(ValueError):
        detector.detect({
            "image": np.zeros((10, 10), dtype=np.uint8)
        })

    with pytest.raises(ValueError):
        DetectionBox(10, 0, 5, 20, 0.9)
