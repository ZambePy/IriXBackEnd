"""Wire contracts for the FastAPI + WebSocket surface (Sprint 12).

Everything a client (frontend) exchanges with the backend is defined here
as a Pydantic model. The models are a *translation layer* on top of
:mod:`irisflow.core.events` — never the other way around — so that the
pipeline can evolve without breaking the wire.

Design rules (SPRINTS §11.3):

* All server → client messages carry a discriminating ``type`` field so
  the frontend can pattern-match without inspecting shape.
* Pixel/normalized coordinates keep the same names as the domain event
  fields whenever possible (``px``, ``py``, ``nx``, ``ny``, ``fixation``).
* Timestamps are always seconds (float, monotonic when possible) — the
  frontend is not expected to line them up against wall time.
* Client → server messages use the same discriminator so a single WS
  message handler can dispatch by ``type``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PROTOCOL_VERSION",
    "CalibrationAbortMessage",
    "CalibrationPointMessage",
    "CalibrationReadyMessage",
    "CalibrationResultMessage",
    "CalibrationStatusResponse",
    "ClickMessage",
    "ClientMessage",
    "ConfigResponse",
    "DwellProgressMessage",
    "ErrorMessage",
    "FaceLostMessage",
    "GazeMessage",
    "HealthResponse",
    "PauseMessage",
    "ProfileInfo",
    "ProfileListResponse",
    "ResumeMessage",
    "SafetyMessage",
    "ScreenInfoPayload",
    "ServerMessage",
    "SetCursorControlMessage",
    "SetProfileMessage",
    "StartCalibrationMessage",
    "StartTrackingMessage",
    "StateMessage",
]


PROTOCOL_VERSION = 1


class _WireBase(BaseModel):
    """Base with forbid-extra so unknown fields on the wire fail loud."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# HTTP responses
# ---------------------------------------------------------------------------
class HealthResponse(_WireBase):
    """``GET /health`` — snapshot for probes and the frontend banner."""

    status: Literal["ok", "starting", "degraded"] = "ok"
    protocol_version: int = PROTOCOL_VERSION
    pipeline_state: str
    backend: str
    fps: float = 0.0
    frames_ok: int = 0
    frames_dropped: int = 0
    frames_face_lost: int = 0


class ConfigResponse(_WireBase):
    """``GET /config`` — the effective config as a JSON tree.

    We return ``AppConfig.model_dump(mode='json')`` verbatim (paths as
    strings). Frontend gets everything; it can pick and choose.
    """

    protocol_version: int = PROTOCOL_VERSION
    config: dict[str, Any]


class ProfileInfo(_WireBase):
    profile_id: str
    screen_width: int
    screen_height: int
    mean_error_px: float
    p95_error_px: float
    n_samples: int
    created_at_wall_s: float


class ProfileListResponse(_WireBase):
    """``GET /profiles`` — every profile visible in ``profiles_dir``."""

    profiles: list[ProfileInfo] = Field(default_factory=list)


class CalibrationStatusResponse(_WireBase):
    """``GET /calibration/status`` — read model for calibration polling."""

    active: bool
    phase: str = "idle"
    current_index: int | None = None
    total: int = 0
    collected: int = 0
    profile_id: str | None = None
    last_error_px: float | None = None


# ---------------------------------------------------------------------------
# Server → Client (WebSocket)
# ---------------------------------------------------------------------------
class GazeMessage(_WireBase):
    """High-frequency gaze position (SPRINTS §11.2)."""

    type: Literal["gaze"] = "gaze"
    frame_id: int
    ts: float
    px: int
    py: int
    nx: float
    ny: float
    fixation: bool
    confidence: float


class StateMessage(_WireBase):
    type: Literal["state"] = "state"
    ts: float
    state: str
    previous: str


class FaceLostMessage(_WireBase):
    type: Literal["face_lost"] = "face_lost"
    ts: float
    duration_ms: float


class DwellProgressMessage(_WireBase):
    type: Literal["dwell_progress"] = "dwell_progress"
    ts: float
    frame_id: int
    px: int
    py: int
    progress: float
    radius_px: int


class ClickMessage(_WireBase):
    type: Literal["click"] = "click"
    ts: float
    frame_id: int
    px: int
    py: int
    button: str = "left"


class CalibrationPointMessage(_WireBase):
    type: Literal["calibration_point"] = "calibration_point"
    ts: float
    index: int
    total: int
    x: float
    y: float
    phase: str


class CalibrationResultMessage(_WireBase):
    type: Literal["calibration_result"] = "calibration_result"
    ts: float
    profile_id: str
    error_px: float
    worst_points: list[int] = Field(default_factory=list)
    accepted: bool


class SafetyMessage(_WireBase):
    type: Literal["safety"] = "safety"
    ts: float
    paused: bool
    reason: str | None = None


class ErrorMessage(_WireBase):
    type: Literal["error"] = "error"
    code: str
    message: str


ServerMessage = Annotated[
    GazeMessage
    | StateMessage
    | FaceLostMessage
    | DwellProgressMessage
    | ClickMessage
    | CalibrationPointMessage
    | CalibrationResultMessage
    | SafetyMessage
    | ErrorMessage,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Client → Server (WebSocket)
# ---------------------------------------------------------------------------
class StartTrackingMessage(_WireBase):
    type: Literal["start_tracking"] = "start_tracking"


class PauseMessage(_WireBase):
    type: Literal["pause"] = "pause"


class ResumeMessage(_WireBase):
    type: Literal["resume"] = "resume"


class ScreenInfoPayload(_WireBase):
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class StartCalibrationMessage(_WireBase):
    type: Literal["start_calibration"] = "start_calibration"
    points: Literal[5, 9, 13] = 9
    screen: ScreenInfoPayload
    profile_id: str = Field(min_length=1, max_length=64)


class CalibrationReadyMessage(_WireBase):
    """Client signals it has displayed target ``index`` — start collecting."""

    type: Literal["calibration_ready"] = "calibration_ready"
    index: int = Field(ge=0)


class CalibrationAbortMessage(_WireBase):
    type: Literal["calibration_abort"] = "calibration_abort"


class SetProfileMessage(_WireBase):
    type: Literal["set_profile"] = "set_profile"
    profile: str = Field(min_length=1, max_length=64)


class SetCursorControlMessage(_WireBase):
    type: Literal["set_cursor_control"] = "set_cursor_control"
    enabled: bool


ClientMessage = Annotated[
    StartTrackingMessage
    | PauseMessage
    | ResumeMessage
    | StartCalibrationMessage
    | CalibrationReadyMessage
    | CalibrationAbortMessage
    | SetProfileMessage
    | SetCursorControlMessage,
    Field(discriminator="type"),
]
