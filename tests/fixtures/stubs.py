"""Test doubles for every Protocol in ``irisflow.core.interfaces``.

The rule (SPRINTS.md instruction 6): these stubs live only under ``tests/``
and are never imported by ``src/irisflow/``. ``import-linter`` enforces this
via the ``production code must not import test stubs`` contract.

Each stub is deterministic — no wall clock, no I/O, no threads — so the
tests that use them stay reproducible in CI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from irisflow.core.types import (
    BoundingBox,
    CalibratedGaze,
    FaceDetection,
    Frame,
    ModelInput,
    ModelMetadata,
    RawGaze,
    ScreenPoint,
    SmoothedPoint,
)

__all__ = [
    "IdentityCalibrationModel",
    "IdentityFilter",
    "IdentityScreenMapper",
    "NoOpCursorController",
    "PassthroughPreprocessor",
    "StubFaceDetector",
    "StubGazeEstimator",
    "SyntheticFrameSource",
]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
@dataclass
class SyntheticFrameSource:
    """Yields deterministic frames of a fixed size, indefinitely."""

    width: int = 640
    height: int = 480
    fps: float = 30.0
    _next_id: int = 0
    _open: bool = False

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def read(self) -> Frame | None:
        if not self._open:
            return None
        data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Distinct per-frame content so tests can tell frames apart.
        data[..., 0] = self._next_id % 255
        frame = Frame(
            data=data,
            frame_id=self._next_id,
            timestamp=self._next_id / self.fps,
        )
        self._next_id += 1
        return frame

    @property
    def is_open(self) -> bool:
        return self._open


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass
class StubFaceDetector:
    """Returns a fixed ``FaceDetection``, or ``None`` on flagged frame ids."""

    face_bbox: BoundingBox = field(default_factory=lambda: BoundingBox(x=100, y=100, w=200, h=200))
    left_eye_bbox: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=120, y=150, w=40, h=25)
    )
    right_eye_bbox: BoundingBox = field(
        default_factory=lambda: BoundingBox(x=220, y=150, w=40, h=25)
    )
    lost_frame_ids: frozenset[int] = field(default_factory=frozenset)

    def detect(self, frame: Frame) -> FaceDetection | None:
        if frame.frame_id in self.lost_frame_ids:
            return None
        return FaceDetection(
            face_bbox=self.face_bbox,
            left_eye_bbox=self.left_eye_bbox,
            right_eye_bbox=self.right_eye_bbox,
            landmarks=np.zeros((478, 3), dtype=np.float32),
            confidence=1.0,
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@dataclass
class PassthroughPreprocessor:
    """Builds a valid-shape ``ModelInput`` filled with zeros — good enough for
    stages downstream that only care about type conformance."""

    face_size: tuple[int, int] = (224, 224)
    eye_size: tuple[int, int] = (112, 112)

    def build(self, frame: Frame, detection: FaceDetection) -> ModelInput:
        fh, fw = self.face_size
        eh, ew = self.eye_size
        return ModelInput(
            face=np.zeros((fh, fw, 3), dtype=np.float32),
            left_eye=np.zeros((eh, ew, 3), dtype=np.float32),
            right_eye=np.zeros((eh, ew, 3), dtype=np.float32),
            rect=np.zeros(12, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@dataclass
class StubGazeEstimator:
    """Emits a scripted trajectory. Never loads a real model — this is the
    dublê referenced by SPRINTS.md §7 that must never leak into ``src/``."""

    trajectory: Sequence[tuple[float, float]] = field(default_factory=lambda: [(0.5, 0.5)])
    inference_ms: float = 5.0
    _cursor: int = 0

    def predict(self, model_input: ModelInput) -> RawGaze:
        x, y = self.trajectory[self._cursor % len(self.trajectory)]
        self._cursor += 1
        return RawGaze(x=x, y=y, confidence=1.0, inference_ms=self.inference_ms)

    def warmup(self) -> None:
        return None

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            backend="stub",
            path="<in-memory>",
            input_names=("face", "left_eye", "right_eye", "rect"),
            face_shape=(224, 224, 3),
            eye_shape=(112, 112, 3),
            rect_dim=12,
            channel_order="RGB",
            normalization="unit",
            output_kind="gaze_xy",
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class IdentityCalibrationModel:
    """No-op calibration — used to test the plumbing without a fit step."""

    profile_id: str = "identity"
    _fitted: bool = True

    def fit(
        self,
        raw_samples: Sequence[RawGaze],
        targets: Sequence[tuple[float, float]],
    ) -> None:
        if len(raw_samples) != len(targets):
            raise ValueError(
                f"fit: len(raw_samples)={len(raw_samples)} != len(targets)={len(targets)}"
            )
        self._fitted = True

    def transform(self, raw: RawGaze) -> CalibratedGaze:
        return CalibratedGaze(x=raw.x, y=raw.y, profile_id=self.profile_id)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
@dataclass
class IdentityScreenMapper:
    """Maps normalized gaze straight to pixels on a fixed screen."""

    screen_width: int = 1920
    screen_height: int = 1080
    screen_id: int = 0

    def to_screen(self, gaze: CalibratedGaze) -> ScreenPoint:
        return ScreenPoint(
            px=round(gaze.x * self.screen_width),
            py=round(gaze.y * self.screen_height),
            screen_id=self.screen_id,
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
@dataclass
class IdentityFilter:
    """Passes ``point`` through unchanged; flags fixation when identical to last."""

    _last: ScreenPoint | None = None

    def apply(self, point: ScreenPoint, timestamp: float) -> SmoothedPoint:
        is_fixation = (
            self._last is not None and self._last.px == point.px and self._last.py == point.py
        )
        self._last = point
        return SmoothedPoint(px=point.px, py=point.py, is_fixation=is_fixation, velocity=0.0)

    def reset(self) -> None:
        self._last = None


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
@dataclass
class NoOpCursorController:
    """Records every call; never touches the OS."""

    moves: list[tuple[int, int]] = field(default_factory=list)
    clicks: list[str] = field(default_factory=list)
    _enabled: bool = False

    def move(self, px: int, py: int) -> None:
        self.moves.append((px, py))

    def click(self, button: str = "left") -> None:
        self.clicks.append(button)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled


def _make_landmarks() -> NDArray[np.float32]:
    """Small helper for tests that need a valid landmarks array."""
    return np.zeros((478, 3), dtype=np.float32)
