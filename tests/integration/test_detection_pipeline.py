"""Sprint 4 integration — detection layer talks to the rest of the system.

Verifies:

* Both the raw and the tracked detectors satisfy the ``FaceDetector``
  protocol structurally.
* A synthetic capture → detector chain runs frame after frame without
  raising, even when the backend flips between "face present" and "face
  missing" states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle

import numpy as np
from numpy.typing import NDArray

from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.core.interfaces import FaceDetector
from irisflow.core.types import Frame
from irisflow.detection.landmarks import (
    FACE_LANDMARK_COUNT_WITH_IRIS,
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from irisflow.detection.mediapipe_detector import MediaPipeFaceDetector
from irisflow.detection.tracking import TrackedFaceDetector


@dataclass
class _AlternatingFaceMesh:
    """Emits landmarks on even calls, nothing on odd calls."""

    landmarks: NDArray[np.float32]
    close_count: int = 0
    _step: int = field(default=0)

    def detect(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        self._step += 1
        return self.landmarks if self._step % 2 == 1 else None

    def close(self) -> None:
        self.close_count += 1


def _make_landmarks() -> NDArray[np.float32]:
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.3, 0.7, size=FACE_LANDMARK_COUNT_WITH_IRIS).astype(np.float32)
    ys = rng.uniform(0.3, 0.7, size=FACE_LANDMARK_COUNT_WITH_IRIS).astype(np.float32)
    zs = np.zeros(FACE_LANDMARK_COUNT_WITH_IRIS, dtype=np.float32)
    for idx in FACE_OVAL_INDICES:
        xs[idx], ys[idx] = 0.5, 0.5
    for idx in LEFT_EYE_INDICES:
        xs[idx], ys[idx] = 0.42, 0.45
    for idx in RIGHT_EYE_INDICES:
        xs[idx], ys[idx] = 0.58, 0.45
    return np.stack([xs, ys, zs], axis=1)


def test_detectors_satisfy_face_detector_protocol() -> None:
    mesh = _AlternatingFaceMesh(landmarks=_make_landmarks())
    raw = MediaPipeFaceDetector(mesh)
    tracked = TrackedFaceDetector(raw)
    assert isinstance(raw, FaceDetector)
    assert isinstance(tracked, FaceDetector)


def test_tracked_pipeline_survives_alternating_absence() -> None:
    mesh = _AlternatingFaceMesh(landmarks=_make_landmarks())
    detector = MediaPipeFaceDetector(mesh, roi_margin=0.1)
    tracked = TrackedFaceDetector(detector, smoothing_alpha=0.5, lost_hysteresis=2)
    source = SyntheticFrameSource(width=200, height=200)
    outputs: list[bool] = []
    with source:
        for _ in range(10):
            frame = source.read()
            assert isinstance(frame, Frame)
            out = tracked.detect(frame)
            outputs.append(out is not None)
    # With hysteresis>=1, every odd frame's miss is held → all outputs present.
    assert all(outputs)


def test_face_absence_without_hysteresis_reports_none() -> None:
    calls = cycle([None])

    class _Silent:
        def detect(self, _: NDArray[np.uint8]) -> NDArray[np.float32] | None:
            return next(calls)

        def close(self) -> None:
            return None

    detector = MediaPipeFaceDetector(_Silent())
    tracked = TrackedFaceDetector(detector, lost_hysteresis=0)
    source = SyntheticFrameSource(width=64, height=64)
    with source:
        for _ in range(5):
            frame = source.read()
            assert frame is not None
            assert tracked.detect(frame) is None
