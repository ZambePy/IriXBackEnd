"""Sprint 11 — round-trip the session file format."""

from __future__ import annotations

from pathlib import Path

import pytest

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
from irisflow.telemetry.session import (
    SCHEMA_VERSION,
    SessionHeader,
    SessionRecordError,
    event_from_line,
    event_to_line,
    header_from_line,
    header_to_line,
    read_session_file,
    write_session_file,
)


def _sample_header() -> SessionHeader:
    return SessionHeader(
        session_id="session-test",
        wall_start_s=1234.5,
        screen_width_px=1920,
        screen_height_px=1080,
        calibration_profile_id="maria",
        config_snapshot={"filtering": {"chain": ["outlier"]}},
    )


def _all_event_kinds() -> list[Event]:
    return [
        RawGazeReady(frame_id=1, timestamp=0.1, x=0.4, y=0.6, confidence=0.9, inference_ms=4.2),
        GazeUpdated(frame_id=1, timestamp=0.1, px=770, py=650, is_fixation=True, confidence=0.9),
        FaceLost(frame_id=2, timestamp=0.2, duration_ms=17.0),
        FaceAcquired(frame_id=3, timestamp=0.3),
        StateChanged(previous=PipelineState.TRACKING, current=PipelineState.LOST, timestamp=0.4),
        CalibrationProgress(index=3, total=9, target_x=0.5, target_y=0.5, phase="collecting"),
        DwellProgress(frame_id=4, timestamp=0.5, px=100, py=200, progress=0.7, radius_px=40),
        DwellClick(frame_id=5, timestamp=0.6, px=100, py=200, button="left"),
        SafetyPaused(timestamp=0.7, reason="kill_switch"),
        SafetyResumed(timestamp=0.8),
    ]


def test_header_round_trip() -> None:
    header = _sample_header()
    round_tripped = header_from_line(header_to_line(header))
    assert round_tripped == header


def test_event_round_trip_all_kinds() -> None:
    for event in _all_event_kinds():
        line = event_to_line(event)
        recovered = event_from_line(line)
        assert type(recovered) is type(event)
        assert recovered == event


def test_write_and_read_session_file(tmp_path: Path) -> None:
    header = _sample_header()
    events = _all_event_kinds()
    path = tmp_path / "recording.jsonl"
    write_session_file(path, header, events)
    round_header, round_events = read_session_file(path)
    assert round_header == header
    assert round_events == events


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SessionRecordError, match="not found"):
        read_session_file(tmp_path / "nope.jsonl")


def test_read_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SessionRecordError, match="empty"):
        read_session_file(path)


def test_read_bad_schema_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "header", "schema_version": 999}\n', encoding="utf-8")
    with pytest.raises(SessionRecordError, match="schema_version"):
        read_session_file(path)


def test_read_missing_kind_raises() -> None:
    with pytest.raises(SessionRecordError, match="kind"):
        header_from_line('{"schema_version": 1}')


def test_write_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "sub" / "rec.jsonl"
    write_session_file(path, _sample_header(), [])
    assert path.exists()


def test_event_from_line_rejects_non_event() -> None:
    with pytest.raises(SessionRecordError, match="expected event"):
        event_from_line('{"kind": "header"}')


def test_current_schema_version_is_one() -> None:
    assert SCHEMA_VERSION == 1
