"""Sprint 6 — batch-dim helper and defensive gaze clamp."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.core.types import ModelInput
from irisflow.inference.base import add_batch_dim, clamp_gaze


def _model_input() -> ModelInput:
    return ModelInput(
        face=np.zeros((224, 224, 3), dtype=np.float32),
        left_eye=np.zeros((112, 112, 3), dtype=np.float32),
        right_eye=np.zeros((112, 112, 3), dtype=np.float32),
        rect=np.zeros(12, dtype=np.float32),
    )


def test_add_batch_dim_prepends_leading_axis_to_every_input() -> None:
    batched = add_batch_dim(_model_input())
    assert batched["face"].shape == (1, 224, 224, 3)
    assert batched["left_eye"].shape == (1, 112, 112, 3)
    assert batched["right_eye"].shape == (1, 112, 112, 3)
    assert batched["rect"].shape == (1, 12)


def test_add_batch_dim_returns_views_not_copies() -> None:
    mi = _model_input()
    batched = add_batch_dim(mi)
    assert batched["face"].base is mi.face
    assert batched["rect"].base is mi.rect


def test_clamp_gaze_passthrough_when_in_range() -> None:
    raw = clamp_gaze(0.3, 0.7, inference_ms=5.0)
    assert raw.x == pytest.approx(0.3)
    assert raw.y == pytest.approx(0.7)
    assert raw.inference_ms == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("x", "y", "clamped_x", "clamped_y"),
    [
        (-0.5, 0.5, 0.0, 0.5),
        (0.5, -0.1, 0.5, 0.0),
        (1.5, 0.5, 1.0, 0.5),
        (0.5, 1.2, 0.5, 1.0),
        (2.0, -3.0, 1.0, 0.0),
    ],
)
def test_clamp_gaze_clips_extrapolations(
    x: float, y: float, clamped_x: float, clamped_y: float
) -> None:
    raw = clamp_gaze(x, y, inference_ms=1.0)
    assert raw.x == pytest.approx(clamped_x)
    assert raw.y == pytest.approx(clamped_y)


def test_clamp_gaze_records_confidence_and_ms() -> None:
    raw = clamp_gaze(0.5, 0.5, inference_ms=12.34, confidence=0.9, backend="fake")
    assert raw.confidence == pytest.approx(0.9)
    assert raw.inference_ms == pytest.approx(12.34)
