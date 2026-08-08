"""Session file format shared by :mod:`recorder`, :mod:`replayer` and
:mod:`report` (Sprint 11).

On-disk layout: one JSON object per line (JSONL). The first line is
always the session header; every subsequent line carries one event.
Newline-delimited JSON is chosen over binary formats so a recording can
be inspected with ``less`` or diffed in code review — the sessions are
small (~30 events/sec, a handful of floats each) and human readability
is worth more than compression.

Determinism promise: the replayer only needs the header + the
``RawGazeReady`` / ``FaceLost`` / ``FaceAcquired`` / ``StateChanged``
events to rebuild every downstream signal (``GazeUpdated``,
``DwellProgress``, ``DwellClick``, ``SafetyPaused``, ``SafetyResumed``).
The other event types are recorded for comparison and for the report,
not for reconstruction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from irisflow.core.events import (
    CalibrationProgress,
    DwellClick,
    DwellProgress,
    Event,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    PipelineState,
    RawGazeReady,
    SafetyPaused,
    SafetyResumed,
    StateChanged,
)

__all__ = [
    "SCHEMA_VERSION",
    "SessionHeader",
    "SessionRecordError",
    "event_from_line",
    "event_to_line",
    "header_from_line",
    "header_to_line",
    "iter_session_events",
    "read_session_file",
    "write_session_file",
]


SCHEMA_VERSION = 1


class SessionRecordError(Exception):
    """Session file is missing, malformed, or from an unsupported schema."""


@dataclass(frozen=True, slots=True)
class SessionHeader:
    """Everything the replayer needs to rebuild the downstream pipeline.

    We store a Pydantic-serialisable dict for the fields the replay
    reproduces (filtering, mapping, control). Calibration is referenced
    by profile id — the actual coefficients live on disk and are loaded
    exactly the same way as in a normal run.
    """

    session_id: str
    wall_start_s: float
    screen_width_px: int
    screen_height_px: int
    calibration_profile_id: str | None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# On-disk envelope
# ---------------------------------------------------------------------------
class _Envelope(TypedDict, total=False):
    kind: str
    type: str


def header_to_line(header: SessionHeader) -> str:
    """Serialise the header as the first line of the session file."""
    payload = {
        "kind": "header",
        "schema_version": header.schema_version,
        "session_id": header.session_id,
        "wall_start_s": header.wall_start_s,
        "screen_width_px": header.screen_width_px,
        "screen_height_px": header.screen_height_px,
        "calibration_profile_id": header.calibration_profile_id,
        "config": header.config_snapshot,
    }
    return json.dumps(payload, sort_keys=True)


def header_from_line(line: str) -> SessionHeader:
    data = _parse_line(line)
    if data.get("kind") != "header":
        raise SessionRecordError(
            f"first line must be a header, got kind={data.get('kind')!r}"
        )
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SessionRecordError(
            f"unsupported session schema_version={version!r} (expected {SCHEMA_VERSION})"
        )
    try:
        return SessionHeader(
            session_id=str(data["session_id"]),
            wall_start_s=float(data["wall_start_s"]),
            screen_width_px=int(data["screen_width_px"]),
            screen_height_px=int(data["screen_height_px"]),
            calibration_profile_id=data.get("calibration_profile_id"),
            config_snapshot=dict(data.get("config") or {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionRecordError(f"header missing/invalid field: {exc}") from exc


def event_to_line(event: Event) -> str:
    """Serialise ``event`` into a JSONL line."""
    payload = _event_to_dict(event)
    payload["kind"] = "event"
    return json.dumps(payload, sort_keys=True)


def event_from_line(line: str) -> Event:
    data = _parse_line(line)
    if data.get("kind") != "event":
        raise SessionRecordError(f"expected event line, got kind={data.get('kind')!r}")
    return _event_from_dict(data)


def write_session_file(
    path: Path,
    header: SessionHeader,
    events: Iterable[Event],
) -> None:
    """One-shot write: header + every event. Overwrites ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(header_to_line(header))
        fh.write("\n")
        for event in events:
            fh.write(event_to_line(event))
            fh.write("\n")


def read_session_file(path: Path) -> tuple[SessionHeader, list[Event]]:
    header, events_iter = iter_session_events(path)
    return header, list(events_iter)


def iter_session_events(path: Path) -> tuple[SessionHeader, Iterator[Event]]:
    """Stream events from ``path``. Header is returned eagerly; events
    lazily via the returned iterator.
    """
    if not path.exists():
        raise SessionRecordError(f"session file not found: {path}")
    fh = path.open("r", encoding="utf-8")
    try:
        first_line = fh.readline()
    except Exception:
        fh.close()
        raise
    if not first_line:
        fh.close()
        raise SessionRecordError(f"session file is empty: {path}")
    header = header_from_line(first_line)

    def _events() -> Iterator[Event]:
        try:
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                yield event_from_line(stripped)
        finally:
            fh.close()

    return header, _events()


# ---------------------------------------------------------------------------
# Event <-> dict
# ---------------------------------------------------------------------------
def _event_to_dict(event: Event) -> dict[str, Any]:
    if isinstance(event, RawGazeReady):
        return {
            "type": "RawGazeReady",
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
            "x": event.x,
            "y": event.y,
            "confidence": event.confidence,
            "inference_ms": event.inference_ms,
        }
    if isinstance(event, GazeUpdated):
        return {
            "type": "GazeUpdated",
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
            "px": event.px,
            "py": event.py,
            "is_fixation": event.is_fixation,
            "confidence": event.confidence,
        }
    if isinstance(event, FaceLost):
        return {
            "type": "FaceLost",
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
            "duration_ms": event.duration_ms,
        }
    if isinstance(event, FaceAcquired):
        return {
            "type": "FaceAcquired",
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
        }
    if isinstance(event, StateChanged):
        return {
            "type": "StateChanged",
            "previous": event.previous.value,
            "current": event.current.value,
            "timestamp": event.timestamp,
        }
    if isinstance(event, CalibrationProgress):
        return {
            "type": "CalibrationProgress",
            "index": event.index,
            "total": event.total,
            "target_x": event.target_x,
            "target_y": event.target_y,
            "phase": event.phase,
        }
    if isinstance(event, DwellProgress):
        return {
            "type": "DwellProgress",
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
            "px": event.px,
            "py": event.py,
            "progress": event.progress,
            "radius_px": event.radius_px,
        }
    if isinstance(event, DwellClick):
        return {
            "type": "DwellClick",
            "frame_id": event.frame_id,
            "timestamp": event.timestamp,
            "px": event.px,
            "py": event.py,
            "button": event.button,
        }
    if isinstance(event, SafetyPaused):
        return {
            "type": "SafetyPaused",
            "timestamp": event.timestamp,
            "reason": event.reason,
        }
    if isinstance(event, SafetyResumed):
        return {
            "type": "SafetyResumed",
            "timestamp": event.timestamp,
        }
    raise SessionRecordError(f"cannot serialise event of type {type(event).__name__}")


def _event_from_dict(data: dict[str, Any]) -> Event:
    event_type = data.get("type")
    if event_type == "RawGazeReady":
        return RawGazeReady(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            x=float(data["x"]),
            y=float(data["y"]),
            confidence=float(data["confidence"]),
            inference_ms=float(data["inference_ms"]),
        )
    if event_type == "GazeUpdated":
        return GazeUpdated(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            px=int(data["px"]),
            py=int(data["py"]),
            is_fixation=bool(data["is_fixation"]),
            confidence=float(data["confidence"]),
        )
    if event_type == "FaceLost":
        return FaceLost(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            duration_ms=float(data["duration_ms"]),
        )
    if event_type == "FaceAcquired":
        return FaceAcquired(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
        )
    if event_type == "StateChanged":
        return StateChanged(
            previous=PipelineState(data["previous"]),
            current=PipelineState(data["current"]),
            timestamp=float(data["timestamp"]),
        )
    if event_type == "CalibrationProgress":
        return CalibrationProgress(
            index=int(data["index"]),
            total=int(data["total"]),
            target_x=float(data["target_x"]),
            target_y=float(data["target_y"]),
            phase=data["phase"],
        )
    if event_type == "DwellProgress":
        return DwellProgress(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            px=int(data["px"]),
            py=int(data["py"]),
            progress=float(data["progress"]),
            radius_px=int(data["radius_px"]),
        )
    if event_type == "DwellClick":
        return DwellClick(
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            px=int(data["px"]),
            py=int(data["py"]),
            button=data.get("button", "left"),
        )
    if event_type == "SafetyPaused":
        return SafetyPaused(
            timestamp=float(data["timestamp"]),
            reason=data["reason"],
        )
    if event_type == "SafetyResumed":
        return SafetyResumed(timestamp=float(data["timestamp"]))
    raise SessionRecordError(f"unknown event type {event_type!r}")


def _parse_line(line: str) -> dict[str, Any]:
    line = line.strip()
    if not line:
        raise SessionRecordError("blank line in session file")
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SessionRecordError(f"line is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise SessionRecordError(f"expected a JSON object, got {type(data).__name__}")
    return data
