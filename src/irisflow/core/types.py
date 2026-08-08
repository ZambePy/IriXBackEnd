"""Immutable domain types passed between pipeline stages.

Every stage receives one of these and returns another one — no ``dict`` or
anonymous tuple ever crosses a stage boundary. This is what kills the
"silent argument-order bug" that plagues vision pipelines.

All types are ``@dataclass(frozen=True, slots=True)``. Types that carry a
numpy array set ``eq=False`` because dataclass-generated ``__eq__`` on an
ndarray raises ``ValueError`` — identity comparison is what we want anyway,
and value comparison of a frame or tensor is almost never a useful operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "BoundingBox",
    "CalibratedGaze",
    "FaceDetection",
    "Frame",
    "ModelInput",
    "ModelMetadata",
    "PipelineTick",
    "RawGaze",
    "ScreenPoint",
    "SmoothedPoint",
]


# ---------------------------------------------------------------------------
# Frame — the raw pixels coming out of the capture thread
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """A single BGR frame with the metadata every downstream stage needs.

    ``timestamp`` is monotonic seconds (``time.monotonic()``), not wall clock,
    so latency deltas remain valid across suspend/resume and NTP jumps.
    """

    data: NDArray[np.uint8]
    frame_id: int
    timestamp: float

    def __post_init__(self) -> None:
        if self.data.ndim != 3 or self.data.shape[2] != 3:
            raise ValueError(f"Frame.data must be HxWx3, got shape {self.data.shape!r}")

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


# ---------------------------------------------------------------------------
# BoundingBox — the workhorse of the detection/preprocessing layers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned integer pixel box.

    Convention: ``(x, y)`` is the top-left corner; ``(w, h)`` are the width
    and height. Zero-sized and negative-position boxes are allowed at
    construction — clipping/expansion helpers are responsible for producing
    valid boxes when needed. This keeps the domain type unopinionated.

    The three transformation helpers are pure — each returns a new box.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def expand(self, margin: float) -> BoundingBox:
        """Grow the box by ``margin`` (fraction of the current side length).

        ``margin = 0.2`` on a 100-px box → 120-px box centered on the same
        point. Result is rounded to integer pixels.
        """
        if margin < 0:
            raise ValueError(f"expand margin must be >= 0, got {margin}")
        dw = round(self.w * margin)
        dh = round(self.h * margin)
        return BoundingBox(
            x=self.x - dw // 2,
            y=self.y - dh // 2,
            w=self.w + dw,
            h=self.h + dh,
        )

    def clip(self, frame_width: int, frame_height: int) -> BoundingBox:
        """Intersect with the frame rectangle. Result may have ``w == 0`` or
        ``h == 0`` if the input lies fully outside the frame — and in that
        case ``(x, y)`` is capped to the frame edge so callers can assume
        ``0 <= x <= frame_width`` and ``0 <= y <= frame_height`` regardless."""
        x0 = min(frame_width, max(0, self.x))
        y0 = min(frame_height, max(0, self.y))
        x1 = min(frame_width, max(0, self.x2))
        y1 = min(frame_height, max(0, self.y2))
        return BoundingBox(x=x0, y=y0, w=max(0, x1 - x0), h=max(0, y1 - y0))

    def to_square(self) -> BoundingBox:
        """Expand the shorter side so ``w == h``, keeping the center fixed.

        A square crop is what the CNN was trained on; distorting the aspect
        ratio by resizing a non-square crop directly is a classic silent bug.
        """
        side = max(self.w, self.h)
        return BoundingBox(
            x=round(self.cx - side / 2.0),
            y=round(self.cy - side / 2.0),
            w=side,
            h=side,
        )

    def normalized(self, frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
        """Return ``(x, y, w, h)`` divided by frame dimensions — used to
        assemble the ``rect`` vector fed to the CNN."""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"frame dims must be positive, got {frame_width}x{frame_height}")
        return (
            self.x / frame_width,
            self.y / frame_height,
            self.w / frame_width,
            self.h / frame_height,
        )


# ---------------------------------------------------------------------------
# FaceDetection — the output of Sprint 4
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class FaceDetection:
    """Landmarks + derived ROIs for a single detected face."""

    face_bbox: BoundingBox
    left_eye_bbox: BoundingBox
    right_eye_bbox: BoundingBox
    landmarks: NDArray[np.float32]
    confidence: float


# ---------------------------------------------------------------------------
# ModelInput — the tensors the CNN actually consumes (Sprint 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class ModelInput:
    """Exact-shape tensors ready for ``GazeEstimator.predict``.

    Shape/dtype are asserted at construction because a shape mismatch here is
    the class of bug that hides behind "the model works but the numbers are
    off" — much cheaper to fail loud in preprocessing than to debug later.
    """

    face: NDArray[np.float32]  # (H_face, W_face, 3)
    left_eye: NDArray[np.float32]  # (H_eye, W_eye, 3)
    right_eye: NDArray[np.float32]  # (H_eye, W_eye, 3)
    rect: NDArray[np.float32]  # (12,)

    def __post_init__(self) -> None:
        for name, expected_ndim, arr in (
            ("face", 3, self.face),
            ("left_eye", 3, self.left_eye),
            ("right_eye", 3, self.right_eye),
        ):
            if arr.ndim != expected_ndim or arr.shape[-1] != 3:
                raise ValueError(f"ModelInput.{name} must be HxWx3, got shape {arr.shape!r}")
            if arr.dtype != np.float32:
                raise ValueError(f"ModelInput.{name} must be float32, got {arr.dtype}")
        if self.rect.shape != (12,) or self.rect.dtype != np.float32:
            raise ValueError(
                f"ModelInput.rect must be (12,) float32, got "
                f"shape={self.rect.shape!r} dtype={self.rect.dtype}"
            )


# ---------------------------------------------------------------------------
# Gaze — raw model output → calibrated → screen pixel → smoothed
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RawGaze:
    """Direct model output, normalized to ``[0, 1]²`` (post-clamp)."""

    x: float
    y: float
    confidence: float
    inference_ms: float


@dataclass(frozen=True, slots=True)
class CalibratedGaze:
    """Raw gaze after the per-user calibration transform."""

    x: float
    y: float
    profile_id: str


@dataclass(frozen=True, slots=True)
class ScreenPoint:
    """Screen-space pixel position; may be produced by mapping or filtering."""

    px: int
    py: int
    screen_id: int = 0


@dataclass(frozen=True, slots=True)
class SmoothedPoint:
    """Filter chain output — the point the cursor should render at."""

    px: int
    py: int
    is_fixation: bool
    velocity: float


# ---------------------------------------------------------------------------
# PipelineTick — one loop iteration, for telemetry and replay
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PipelineTick:
    """Snapshot of one iteration of the main loop.

    ``point`` is ``None`` when the pipeline could not produce a valid gaze
    for this frame (face lost, paused, etc.) — an explicit absence, not a
    stale reused value.
    """

    frame_id: int
    timestamp: float
    state: str
    stage_timings_ms: dict[str, float]
    point: SmoothedPoint | None


# ---------------------------------------------------------------------------
# Model metadata — the answer that Sprint 6 writes into MODEL_CARD.md
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """The empirical contract of the loaded model.

    Written once at load time by the backend and read by preprocessing to
    stay bit-exact with what the model expects. Every field here is a
    hypothesis until confirmed by Sprint 6.
    """

    backend: str
    path: str
    input_names: tuple[str, ...]
    face_shape: tuple[int, int, int]
    eye_shape: tuple[int, int, int]
    rect_dim: int
    channel_order: str
    normalization: str
    output_kind: str
