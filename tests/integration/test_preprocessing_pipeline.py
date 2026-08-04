"""Sprint 5 integration — preprocessing sits on top of detection cleanly.

Verifies the full detection → preprocessing chain produces a valid
:class:`ModelInput` from a synthetic frame + fake FaceMesh backend. If
this ever breaks, either the detection output changed shape or the
builder started rejecting valid inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.detection.landmarks import (
    FACE_LANDMARK_COUNT_WITH_IRIS,
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from irisflow.detection.mediapipe_detector import MediaPipeFaceDetector
from irisflow.preprocessing.builder import ModelInputBuilder


@dataclass
class _StaticMesh:
    landmarks: NDArray[np.float32]

    def detect(self, _: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        return self.landmarks

    def close(self) -> None:
        return None


def _plausible_landmarks() -> NDArray[np.float32]:
    rng = np.random.default_rng(1)
    xs = rng.uniform(0.3, 0.7, size=FACE_LANDMARK_COUNT_WITH_IRIS).astype(np.float32)
    ys = rng.uniform(0.3, 0.7, size=FACE_LANDMARK_COUNT_WITH_IRIS).astype(np.float32)
    zs = np.zeros(FACE_LANDMARK_COUNT_WITH_IRIS, dtype=np.float32)
    # Spread the oval landmarks across the frame so face bbox has area.
    for i, idx in enumerate(FACE_OVAL_INDICES):
        angle = 2 * np.pi * i / len(FACE_OVAL_INDICES)
        xs[idx] = 0.5 + 0.25 * float(np.cos(angle))
        ys[idx] = 0.5 + 0.25 * float(np.sin(angle))
    # Left eye near x=0.42, right near x=0.58 — small ellipse for each.
    for i, idx in enumerate(LEFT_EYE_INDICES):
        angle = 2 * np.pi * i / len(LEFT_EYE_INDICES)
        xs[idx] = 0.42 + 0.03 * float(np.cos(angle))
        ys[idx] = 0.45 + 0.015 * float(np.sin(angle))
    for i, idx in enumerate(RIGHT_EYE_INDICES):
        angle = 2 * np.pi * i / len(RIGHT_EYE_INDICES)
        xs[idx] = 0.58 + 0.03 * float(np.cos(angle))
        ys[idx] = 0.45 + 0.015 * float(np.sin(angle))
    return np.stack([xs, ys, zs], axis=1)


def test_end_to_end_detection_then_preprocessing_produces_model_input() -> None:
    mesh = _StaticMesh(landmarks=_plausible_landmarks())
    detector = MediaPipeFaceDetector(mesh, roi_margin=0.1)
    builder = ModelInputBuilder()
    source = SyntheticFrameSource(width=320, height=240, pattern="gradient")
    with source:
        frame = source.read()
        assert frame is not None
        detection = detector.detect(frame)
        assert detection is not None
        result = builder.build(frame, detection)
    assert result.face.shape == (224, 224, 3)
    assert result.left_eye.shape == (112, 112, 3)
    assert result.right_eye.shape == (112, 112, 3)
    assert result.rect.shape == (12,)
    assert result.rect.min() >= 0.0
    assert result.rect.max() <= 1.0
