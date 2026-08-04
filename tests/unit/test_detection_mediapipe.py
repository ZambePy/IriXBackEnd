"""Sprint 4 — MediaPipe adapter over a stubbed FaceMesh backend.

The real MediaPipe FaceLandmarker needs a ``.task`` model file and a lot
of native code — none of which we want in a fast unit test. Instead we
inject a :class:`FaceMeshLike` fake that emits a known landmark array and
verify that the adapter:

* Returns ``None`` when no face is present.
* Returns ``None`` when the backend raises (never propagates upwards).
* Builds a :class:`FaceDetection` with sensible box coverage otherwise.
* Rejects landmark arrays of unexpected size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from irisflow.core.exceptions import DetectionError
from irisflow.core.types import Frame
from irisflow.detection.landmarks import (
    FACE_LANDMARK_COUNT_WITH_IRIS,
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from irisflow.detection.mediapipe_detector import (
    MediaPipeFaceDetector,
    build_default_face_mesh,
)


@dataclass
class FakeFaceMesh:
    """Deterministic FaceMesh: returns the same landmark array every call."""

    landmarks: NDArray[np.float32] | None
    raise_on_call: Exception | None = None
    close_count: int = 0
    detect_calls: int = 0

    def detect(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        self.detect_calls += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.landmarks

    def close(self) -> None:
        self.close_count += 1


def _synthetic_landmarks(width: int = 100, height: int = 100) -> NDArray[np.float32]:
    """Build a plausible 478-landmark array clustered around the face center."""
    rng = np.random.default_rng(42)
    # Base cloud in the middle two thirds of the frame — matches roughly what
    # a centred face would produce after MediaPipe normalization.
    xs = rng.uniform(0.3, 0.7, size=FACE_LANDMARK_COUNT_WITH_IRIS).astype(np.float32)
    ys = rng.uniform(0.25, 0.75, size=FACE_LANDMARK_COUNT_WITH_IRIS).astype(np.float32)
    zs = np.zeros(FACE_LANDMARK_COUNT_WITH_IRIS, dtype=np.float32)
    # Force known extrema at oval / eye indices so we can predict bbox behaviour.
    for idx in FACE_OVAL_INDICES:
        xs[idx] = np.clip(xs[idx], 0.15, 0.85)
        ys[idx] = np.clip(ys[idx], 0.15, 0.85)
    for idx in LEFT_EYE_INDICES:
        xs[idx] = np.clip(xs[idx], 0.35, 0.45)
        ys[idx] = np.clip(ys[idx], 0.4, 0.5)
    for idx in RIGHT_EYE_INDICES:
        xs[idx] = np.clip(xs[idx], 0.55, 0.65)
        ys[idx] = np.clip(ys[idx], 0.4, 0.5)
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def _make_frame(width: int = 100, height: int = 100) -> Frame:
    return Frame(
        data=np.zeros((height, width, 3), dtype=np.uint8), frame_id=0, timestamp=0.0
    )


# ---------------------------------------------------------------------------
# Detector behaviour
# ---------------------------------------------------------------------------
def test_detector_returns_none_when_backend_returns_none() -> None:
    mesh = FakeFaceMesh(landmarks=None)
    detector = MediaPipeFaceDetector(mesh)
    assert detector.detect(_make_frame()) is None


def test_detector_returns_none_when_backend_raises() -> None:
    mesh = FakeFaceMesh(landmarks=None, raise_on_call=RuntimeError("boom"))
    detector = MediaPipeFaceDetector(mesh)
    assert detector.detect(_make_frame()) is None


def test_detector_returns_face_detection_when_backend_yields_landmarks() -> None:
    mesh = FakeFaceMesh(landmarks=_synthetic_landmarks())
    detector = MediaPipeFaceDetector(mesh, roi_margin=0.0, square_rois=False)
    result = detector.detect(_make_frame())
    assert result is not None
    assert result.face_bbox.w > 0
    assert result.face_bbox.h > 0
    assert result.left_eye_bbox.w > 0
    assert result.right_eye_bbox.w > 0
    # left eye is to the subject's left (image x lower) — in the synthetic
    # data we placed left at x∈[0.35, 0.45] and right at x∈[0.55, 0.65]
    assert result.left_eye_bbox.x < result.right_eye_bbox.x


def test_detector_rejects_unexpected_landmark_count() -> None:
    bogus = np.zeros((17, 3), dtype=np.float32)
    mesh = FakeFaceMesh(landmarks=bogus)
    detector = MediaPipeFaceDetector(mesh)
    assert detector.detect(_make_frame()) is None


def test_detector_boxes_are_squared_when_configured() -> None:
    mesh = FakeFaceMesh(landmarks=_synthetic_landmarks())
    detector = MediaPipeFaceDetector(mesh, square_rois=True)
    result = detector.detect(_make_frame())
    assert result is not None
    for box in (result.face_bbox, result.left_eye_bbox, result.right_eye_bbox):
        # to_square + clip may crop one side at the frame edge, but at
        # construction time before clip the boxes were square.
        assert abs(box.w - box.h) <= 1 or box.x == 0 or box.y == 0 or (
            box.x + box.w >= 100 or box.y + box.h >= 100
        )


def test_detector_clips_to_frame() -> None:
    mesh = FakeFaceMesh(landmarks=_synthetic_landmarks())
    detector = MediaPipeFaceDetector(mesh, roi_margin=0.9, square_rois=True)
    result = detector.detect(_make_frame(width=100, height=100))
    assert result is not None
    for box in (result.face_bbox, result.left_eye_bbox, result.right_eye_bbox):
        assert 0 <= box.x <= 100
        assert 0 <= box.y <= 100
        assert box.x + box.w <= 100
        assert box.y + box.h <= 100


def test_detector_close_delegates_to_backend() -> None:
    mesh = FakeFaceMesh(landmarks=None)
    detector = MediaPipeFaceDetector(mesh)
    detector.close()
    assert mesh.close_count == 1


def test_detector_rejects_invalid_margin() -> None:
    with pytest.raises(ValueError, match="roi_margin"):
        MediaPipeFaceDetector(FakeFaceMesh(landmarks=None), roi_margin=1.5)


def test_detector_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        MediaPipeFaceDetector(FakeFaceMesh(landmarks=None), min_confidence=-0.1)


# ---------------------------------------------------------------------------
# Real-backend factory — only path we can safely exercise without hardware
# is the "model missing" one.
# ---------------------------------------------------------------------------
def test_build_default_face_mesh_raises_when_model_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.task"
    with pytest.raises(DetectionError, match="face landmarker model not found"):
        build_default_face_mesh(model_path=missing)
