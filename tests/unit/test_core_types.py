"""Sprint 2 — domain type invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from irisflow.core.types import (
    BoundingBox,
    CalibratedGaze,
    FaceDetection,
    Frame,
    ModelInput,
    ModelMetadata,
    PipelineTick,
    RawGaze,
    ScreenPoint,
    SmoothedPoint,
)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
def _frame(w: int = 640, h: int = 480, frame_id: int = 0) -> Frame:
    return Frame(data=np.zeros((h, w, 3), dtype=np.uint8), frame_id=frame_id, timestamp=0.0)


def test_frame_exposes_width_and_height_from_data() -> None:
    frame = _frame(w=1280, h=720)
    assert frame.width == 1280
    assert frame.height == 720


def test_frame_rejects_non_3d_data() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        Frame(data=np.zeros((480, 640), dtype=np.uint8), frame_id=0, timestamp=0.0)


def test_frame_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        Frame(data=np.zeros((480, 640, 4), dtype=np.uint8), frame_id=0, timestamp=0.0)


def test_frame_is_frozen() -> None:
    frame = _frame()
    with pytest.raises(FrozenInstanceError):
        frame.frame_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BoundingBox — derived properties (transformation methods are covered in
# test_core_geometry.py, which exercises them via geometry.expand/clip/etc.)
# ---------------------------------------------------------------------------
def test_bounding_box_derived_properties() -> None:
    box = BoundingBox(x=10, y=20, w=30, h=40)
    assert box.x2 == 40
    assert box.y2 == 60
    assert box.cx == 25.0
    assert box.cy == 40.0
    assert box.area == 1200


def test_bounding_box_area_of_degenerate_box_is_zero() -> None:
    assert BoundingBox(0, 0, 0, 5).area == 0
    assert BoundingBox(0, 0, 5, -1).area == 0


# ---------------------------------------------------------------------------
# ModelInput — shape + dtype gates
# ---------------------------------------------------------------------------
def _valid_model_input() -> ModelInput:
    return ModelInput(
        face=np.zeros((224, 224, 3), dtype=np.float32),
        left_eye=np.zeros((112, 112, 3), dtype=np.float32),
        right_eye=np.zeros((112, 112, 3), dtype=np.float32),
        rect=np.zeros(12, dtype=np.float32),
    )


def test_model_input_accepts_expected_shapes() -> None:
    inp = _valid_model_input()
    assert inp.face.shape == (224, 224, 3)
    assert inp.rect.shape == (12,)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("face", np.zeros((224, 224), dtype=np.float32)),
        ("face", np.zeros((224, 224, 4), dtype=np.float32)),
        ("left_eye", np.zeros((112, 112, 3), dtype=np.float64)),
    ],
)
def test_model_input_rejects_wrong_face_or_eye_tensor(field: str, bad: np.ndarray) -> None:
    kwargs = {
        "face": np.zeros((224, 224, 3), dtype=np.float32),
        "left_eye": np.zeros((112, 112, 3), dtype=np.float32),
        "right_eye": np.zeros((112, 112, 3), dtype=np.float32),
        "rect": np.zeros(12, dtype=np.float32),
    }
    kwargs[field] = bad
    with pytest.raises(ValueError, match="ModelInput"):
        ModelInput(**kwargs)


def test_model_input_rejects_wrong_rect_shape() -> None:
    with pytest.raises(ValueError, match="rect"):
        ModelInput(
            face=np.zeros((224, 224, 3), dtype=np.float32),
            left_eye=np.zeros((112, 112, 3), dtype=np.float32),
            right_eye=np.zeros((112, 112, 3), dtype=np.float32),
            rect=np.zeros(11, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Gaze value types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("obj", "attr"),
    [
        (RawGaze(x=0.1, y=0.2, confidence=0.9, inference_ms=5.0), "x"),
        (CalibratedGaze(x=0.1, y=0.2, profile_id="p"), "y"),
        (ScreenPoint(px=100, py=200), "px"),
        (SmoothedPoint(px=100, py=200, is_fixation=False, velocity=1.0), "velocity"),
    ],
)
def test_gaze_types_are_immutable(obj: object, attr: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(obj, attr, 99)


def test_screen_point_defaults_to_screen_id_zero() -> None:
    assert ScreenPoint(px=1, py=2).screen_id == 0


# ---------------------------------------------------------------------------
# PipelineTick and ModelMetadata
# ---------------------------------------------------------------------------
def test_pipeline_tick_carries_optional_point() -> None:
    tick = PipelineTick(
        frame_id=1,
        timestamp=0.0,
        state="TRACKING",
        stage_timings_ms={"detect": 10.0},
        point=None,
    )
    assert tick.point is None
    assert tick.stage_timings_ms == {"detect": 10.0}


def test_model_metadata_is_a_simple_record() -> None:
    md = ModelMetadata(
        backend="keras",
        path="models/x.keras",
        input_names=("face",),
        face_shape=(224, 224, 3),
        eye_shape=(112, 112, 3),
        rect_dim=12,
        channel_order="RGB",
        normalization="unit",
        output_kind="gaze_xy",
    )
    assert md.backend == "keras"
    assert md.input_names == ("face",)


# ---------------------------------------------------------------------------
# FaceDetection sanity — ndarrays present, dataclass frozen
# ---------------------------------------------------------------------------
def test_face_detection_holds_landmarks_and_boxes() -> None:
    detection = FaceDetection(
        face_bbox=BoundingBox(0, 0, 100, 100),
        left_eye_bbox=BoundingBox(10, 20, 20, 10),
        right_eye_bbox=BoundingBox(60, 20, 20, 10),
        landmarks=np.zeros((478, 3), dtype=np.float32),
        confidence=0.9,
    )
    assert detection.landmarks.shape == (478, 3)
    assert detection.face_bbox.w == 100
