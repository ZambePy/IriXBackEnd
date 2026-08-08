"""Events published on the pipeline bus.

Every event is an immutable dataclass; consumers pattern-match on the type
(``isinstance`` or ``match/case``). Downstream sinks (cursor, telemetry,
WebSocket) subscribe to whichever event types they care about.

Event schemas here are the source of truth: the Sprint-12 WebSocket
messages in ``api/schemas.py`` are derived from these, not the other way
around.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "CalibrationPhase",
    "CalibrationProgress",
    "DwellClick",
    "DwellProgress",
    "Event",
    "FaceAcquired",
    "FaceLost",
    "GazeUpdated",
    "PipelineState",
    "RawGazeReady",
    "SafetyPaused",
    "SafetyResumed",
    "StateChanged",
]


class PipelineState(StrEnum):
    """States of the pipeline state machine (SPRINTS.md §4.3)."""

    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    PAUSED = "PAUSED"


CalibrationPhase = Literal["prompt", "collecting", "settled", "done", "aborted"]


@dataclass(frozen=True, slots=True)
class RawGazeReady:
    """Raw model output right after inference — Sprint 7's first fim-a-fim signal.

    Sprint 8 adds calibration, Sprint 9 adds mapping/filtering; once those
    stages exist the pipeline publishes :class:`GazeUpdated` (integer
    pixels + fixation flag) instead. Until then subscribers consume the
    normalized floats directly.
    """

    frame_id: int
    timestamp: float
    x: float
    y: float
    confidence: float
    inference_ms: float


@dataclass(frozen=True, slots=True)
class GazeUpdated:
    """Emitted at ~30 Hz once the pipeline has produced a usable point."""

    frame_id: int
    timestamp: float
    px: int
    py: int
    is_fixation: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class FaceLost:
    """Face was tracked and is now gone. ``duration_ms == 0`` on the first frame."""

    frame_id: int
    timestamp: float
    duration_ms: float


@dataclass(frozen=True, slots=True)
class FaceAcquired:
    """Face was absent and just came back."""

    frame_id: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class StateChanged:
    """State machine transition — always paired (``previous`` != ``current``)."""

    previous: PipelineState
    current: PipelineState
    timestamp: float


@dataclass(frozen=True, slots=True)
class CalibrationProgress:
    """One tick of the calibration session state machine."""

    index: int
    total: int
    target_x: float
    target_y: float
    phase: CalibrationPhase


@dataclass(frozen=True, slots=True)
class DwellProgress:
    """One tick of the dwell-click state machine (Sprint 10).

    ``progress`` is a fraction in ``[0, 1]``: 0 when the pointer just entered
    a fresh dwell cluster, 1 the instant a click fires. Sinks (frontend or
    telemetry) can draw a progress ring straight from this value.
    """

    frame_id: int
    timestamp: float
    px: int
    py: int
    progress: float
    radius_px: int


@dataclass(frozen=True, slots=True)
class DwellClick:
    """A dwell-triggered click at ``(px, py)``. Emitted once per click."""

    frame_id: int
    timestamp: float
    px: int
    py: int
    button: Literal["left", "right", "middle"] = "left"


@dataclass(frozen=True, slots=True)
class SafetyPaused:
    """Safety layer disabled the cursor (kill switch, watchdog, face lost)."""

    timestamp: float
    reason: Literal["kill_switch", "face_lost", "watchdog", "manual"]


@dataclass(frozen=True, slots=True)
class SafetyResumed:
    """Safety layer re-enabled the cursor after a previous pause."""

    timestamp: float


Event = (
    RawGazeReady
    | GazeUpdated
    | FaceLost
    | FaceAcquired
    | StateChanged
    | CalibrationProgress
    | DwellProgress
    | DwellClick
    | SafetyPaused
    | SafetyResumed
)
"""Type alias for anything that can appear on the bus."""
