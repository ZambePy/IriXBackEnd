"""Sprint 2 — every Protocol has a stub that satisfies it structurally.

The isinstance check is runtime-shallow (Python's ``runtime_checkable`` only
verifies attribute presence). That is exactly what we want here: it catches
"forgot to implement a method" errors in the stub or in future concrete
implementations, without asserting exact signatures — which would be
overspecified for a structural interface.
"""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.core.interfaces import (
    CalibrationModel,
    CursorController,
    FaceDetector,
    Filter,
    FrameSource,
    GazeEstimator,
    Preprocessor,
    ScreenMapper,
    iter_face_rois,
)
from irisflow.core.types import BoundingBox, FaceDetection, RawGaze, ScreenPoint
from tests.fixtures.stubs import (
    IdentityCalibrationModel,
    IdentityFilter,
    IdentityScreenMapper,
    NoOpCursorController,
    PassthroughPreprocessor,
    StubFaceDetector,
    StubGazeEstimator,
    SyntheticFrameSource,
)

_PROTOCOL_STUB_PAIRS = [
    (FrameSource, SyntheticFrameSource()),
    (FaceDetector, StubFaceDetector()),
    (Preprocessor, PassthroughPreprocessor()),
    (GazeEstimator, StubGazeEstimator()),
    (CalibrationModel, IdentityCalibrationModel()),
    (ScreenMapper, IdentityScreenMapper()),
    (Filter, IdentityFilter()),
    (CursorController, NoOpCursorController()),
]


@pytest.mark.parametrize(
    ("protocol", "stub"),
    _PROTOCOL_STUB_PAIRS,
    ids=[proto.__name__ for proto, _ in _PROTOCOL_STUB_PAIRS],
)
def test_stub_conforms_to_protocol(protocol: type, stub: object) -> None:
    assert isinstance(stub, protocol)


# ---------------------------------------------------------------------------
# Behavioural sanity of a few stubs — cheap wins that catch obvious breakage.
# ---------------------------------------------------------------------------
def test_synthetic_source_returns_none_when_closed() -> None:
    src = SyntheticFrameSource()
    assert src.read() is None
    src.open()
    assert src.read() is not None
    src.close()
    assert src.read() is None


def test_stub_detector_honors_lost_frame_ids() -> None:
    src = SyntheticFrameSource()
    src.open()
    detector = StubFaceDetector(lost_frame_ids=frozenset({1}))

    frame0 = src.read()
    frame1 = src.read()
    assert frame0 is not None
    assert frame1 is not None
    assert detector.detect(frame0) is not None
    assert detector.detect(frame1) is None


def test_stub_gaze_estimator_walks_the_trajectory() -> None:
    estimator = StubGazeEstimator(trajectory=[(0.0, 0.0), (1.0, 1.0)])
    preproc = PassthroughPreprocessor()
    src = SyntheticFrameSource()
    src.open()
    frame = src.read()
    assert frame is not None
    detection = StubFaceDetector().detect(frame)
    assert detection is not None
    model_input = preproc.build(frame, detection)

    first = estimator.predict(model_input)
    second = estimator.predict(model_input)
    third = estimator.predict(model_input)

    assert (first.x, first.y) == (0.0, 0.0)
    assert (second.x, second.y) == (1.0, 1.0)
    assert (third.x, third.y) == (0.0, 0.0)


def test_identity_screen_mapper_scales_to_screen_dims() -> None:
    mapper = IdentityScreenMapper(screen_width=1000, screen_height=500)
    point = mapper.to_screen(_calibrated(0.5, 0.5))
    assert (point.px, point.py) == (500, 250)


def test_identity_filter_detects_repeat_as_fixation() -> None:
    flt = IdentityFilter()
    first = flt.apply(ScreenPoint(px=10, py=20), timestamp=0.0)
    second = flt.apply(ScreenPoint(px=10, py=20), timestamp=1.0)
    third = flt.apply(ScreenPoint(px=11, py=20), timestamp=2.0)
    assert first.is_fixation is False
    assert second.is_fixation is True
    assert third.is_fixation is False


def test_noop_cursor_controller_records_calls() -> None:
    cursor = NoOpCursorController()
    cursor.enable()
    cursor.move(100, 200)
    cursor.click("right")
    assert cursor.is_enabled is True
    assert cursor.moves == [(100, 200)]
    assert cursor.clicks == ["right"]


def test_identity_calibration_model_reports_fitted_and_rejects_mismatched_fit() -> None:
    model = IdentityCalibrationModel()
    assert model.is_fitted is True
    model.fit([_raw(0.1, 0.2)], [(0.1, 0.2)])
    with pytest.raises(ValueError, match="len"):
        model.fit([_raw(0.1, 0.2)], [(0.1, 0.2), (0.3, 0.4)])
    calibrated = model.transform(_raw(0.4, 0.6))
    assert (calibrated.x, calibrated.y) == (0.4, 0.6)
    assert calibrated.profile_id == "identity"


# ---------------------------------------------------------------------------
# iter_face_rois — convenience helper on core types
# ---------------------------------------------------------------------------
def test_iter_face_rois_yields_face_then_eyes_in_order() -> None:
    detection = FaceDetection(
        face_bbox=BoundingBox(0, 0, 100, 100),
        left_eye_bbox=BoundingBox(10, 20, 30, 15),
        right_eye_bbox=BoundingBox(60, 20, 30, 15),
        landmarks=np.zeros((478, 3), dtype=np.float32),
        confidence=1.0,
    )
    rois = list(iter_face_rois(detection))
    assert rois == [detection.face_bbox, detection.left_eye_bbox, detection.right_eye_bbox]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _raw(x: float, y: float) -> RawGaze:
    return RawGaze(x=x, y=y, confidence=1.0, inference_ms=1.0)


def _calibrated(x: float, y: float):
    from irisflow.core.types import CalibratedGaze

    return CalibratedGaze(x=x, y=y, profile_id="p")
