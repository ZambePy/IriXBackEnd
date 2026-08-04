"""Structural interfaces (``Protocol``s) at every seam of the pipeline.

The rule is: no domain code depends on a concrete adapter — everyone talks
through one of these. Swapping MediaPipe for another detector, or Keras
for ONNX, becomes a one-line change in the orchestrator plus a new
implementation of the relevant Protocol.

Marked ``runtime_checkable`` so tests can assert conformance via
``isinstance(stub, Protocol)`` (structural, not nominal). The check is
shallow — it only verifies attribute presence, not signatures — which is
still enough to catch "forgot to implement" bugs in stubs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

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
    "CalibrationModel",
    "CursorController",
    "FaceDetector",
    "Filter",
    "FrameSource",
    "GazeEstimator",
    "Preprocessor",
    "ScreenMapper",
]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
@runtime_checkable
class FrameSource(Protocol):
    """Anything that can hand out the most recent frame with a timestamp.

    Implementations own their own thread/lifetime. ``read`` may return
    ``None`` when the source is temporarily unavailable — that is a normal
    condition, not an error (the pipeline enters ``LOST``).
    """

    def open(self) -> None: ...
    def close(self) -> None: ...
    def read(self) -> Frame | None: ...

    @property
    def is_open(self) -> bool: ...


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@runtime_checkable
class FaceDetector(Protocol):
    """Frame → landmarks + face/eye ROIs. ``None`` means "no face this frame"."""

    def detect(self, frame: Frame) -> FaceDetection | None: ...


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@runtime_checkable
class Preprocessor(Protocol):
    """Assemble the exact tensors the CNN was trained on."""

    def build(self, frame: Frame, detection: FaceDetection) -> ModelInput: ...


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@runtime_checkable
class GazeEstimator(Protocol):
    """The one and only motor of gaze estimation (SPRINTS.md §7.5).

    ``metadata`` is a property, not a method, so consumers can rely on it
    being available immediately after construction — before any predict call.
    """

    def predict(self, model_input: ModelInput) -> RawGaze: ...
    def warmup(self) -> None: ...

    @property
    def metadata(self) -> ModelMetadata: ...


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@runtime_checkable
class CalibrationModel(Protocol):
    """Corrects the per-user bias between raw gaze and true screen intent."""

    def fit(
        self,
        raw_samples: Sequence[RawGaze],
        targets: Sequence[tuple[float, float]],
    ) -> None: ...
    def transform(self, raw: RawGaze) -> CalibratedGaze: ...

    @property
    def is_fitted(self) -> bool: ...


# ---------------------------------------------------------------------------
# Mapping — calibrated normalized → screen pixel
# ---------------------------------------------------------------------------
@runtime_checkable
class ScreenMapper(Protocol):
    """Normalized calibrated gaze → integer screen coordinate."""

    def to_screen(self, gaze: CalibratedGaze) -> ScreenPoint: ...


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
@runtime_checkable
class Filter(Protocol):
    """Temporal filter on the screen-space signal.

    ``apply`` receives ``timestamp`` explicitly so filters that depend on
    inter-sample dt (One Euro) don't need to peek at a clock — the caller
    already has the frame timestamp.
    """

    def apply(self, point: ScreenPoint, timestamp: float) -> SmoothedPoint: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Control (side-effecting; only real implementations touch the OS)
# ---------------------------------------------------------------------------
@runtime_checkable
class CursorController(Protocol):
    """Moves the OS cursor and issues clicks. Safety wrappers live above."""

    def move(self, px: int, py: int) -> None: ...
    def click(self, button: str = "left") -> None: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...

    @property
    def is_enabled(self) -> bool: ...


# ---------------------------------------------------------------------------
# Helper: expose the ROIs of a FaceDetection as an ordered iterable.
# Not a Protocol — just a small utility that belongs here because it works
# entirely on core types and is used by several stages.
# ---------------------------------------------------------------------------
def iter_face_rois(detection: FaceDetection) -> Iterable[BoundingBox]:
    """Yield ``(face, left_eye, right_eye)`` boxes in the canonical order."""
    yield detection.face_bbox
    yield detection.left_eye_bbox
    yield detection.right_eye_bbox
