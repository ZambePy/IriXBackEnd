"""Sprint 11 — SessionRecorder as a bus sink."""

from __future__ import annotations

from pathlib import Path

from irisflow.core.events import (
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    RawGazeReady,
    SafetyPaused,
)
from irisflow.pipeline.bus import EventBus
from irisflow.telemetry.recorder import SessionRecorder
from irisflow.telemetry.session import SessionHeader, read_session_file


def _header() -> SessionHeader:
    return SessionHeader(
        session_id="rec-test",
        wall_start_s=0.0,
        screen_width_px=800,
        screen_height_px=600,
        calibration_profile_id=None,
    )


def _recorder(path: Path, bus: EventBus) -> SessionRecorder:
    return SessionRecorder(
        output_path=path,
        header=_header(),
        subscribe=bus.subscribe,
        unsubscribe=bus.unsubscribe,
    )


def test_recorder_writes_header_only_when_no_events(tmp_path: Path) -> None:
    bus = EventBus()
    rec = _recorder(tmp_path / "s.jsonl", bus)
    rec.start()
    rec.stop()
    header, events = read_session_file(rec.output_path)
    assert header == _header()
    assert events == []


def test_recorder_captures_all_relevant_events(tmp_path: Path) -> None:
    bus = EventBus()
    rec = _recorder(tmp_path / "s.jsonl", bus)
    rec.start()

    bus.publish(
        RawGazeReady(frame_id=1, timestamp=0.1, x=0.5, y=0.5, confidence=1.0, inference_ms=3.0)
    )
    bus.publish(
        GazeUpdated(frame_id=1, timestamp=0.1, px=400, py=300, is_fixation=False, confidence=1.0)
    )
    bus.publish(FaceLost(frame_id=2, timestamp=0.2, duration_ms=0.0))
    bus.publish(FaceAcquired(frame_id=3, timestamp=0.3))
    bus.publish(SafetyPaused(timestamp=0.4, reason="kill_switch"))

    rec.stop()
    _, events = read_session_file(rec.output_path)
    assert len(events) == 5
    assert isinstance(events[0], RawGazeReady)
    assert isinstance(events[-1], SafetyPaused)


def test_recorder_unsubscribes_on_stop(tmp_path: Path) -> None:
    bus = EventBus()
    rec = _recorder(tmp_path / "s.jsonl", bus)
    rec.start()
    bus.publish(
        RawGazeReady(frame_id=1, timestamp=0.0, x=0.5, y=0.5, confidence=1.0, inference_ms=0.0)
    )
    rec.stop()
    # After stop, further events must not land in the file.
    bus.publish(
        RawGazeReady(frame_id=2, timestamp=0.5, x=0.1, y=0.1, confidence=1.0, inference_ms=0.0)
    )
    _, events = read_session_file(rec.output_path)
    assert len(events) == 1


def test_recorder_is_context_manager(tmp_path: Path) -> None:
    bus = EventBus()
    with _recorder(tmp_path / "s.jsonl", bus) as rec:
        bus.publish(
            RawGazeReady(frame_id=1, timestamp=0.0, x=0.5, y=0.5, confidence=1.0, inference_ms=0.0)
        )
    assert not rec.is_started
    _, events = read_session_file(rec.output_path)
    assert len(events) == 1


def test_recorder_start_is_idempotent(tmp_path: Path) -> None:
    bus = EventBus()
    rec = _recorder(tmp_path / "s.jsonl", bus)
    rec.start()
    rec.start()  # no crash, no duplicate subscription
    bus.publish(
        RawGazeReady(frame_id=1, timestamp=0.0, x=0.5, y=0.5, confidence=1.0, inference_ms=0.0)
    )
    rec.stop()
    _, events = read_session_file(rec.output_path)
    assert len(events) == 1


def test_recorder_creates_parent_dir(tmp_path: Path) -> None:
    bus = EventBus()
    target = tmp_path / "deep" / "nested" / "s.jsonl"
    rec = _recorder(target, bus)
    rec.start()
    rec.stop()
    assert target.exists()


def test_recorder_counts_events_written(tmp_path: Path) -> None:
    bus = EventBus()
    rec = _recorder(tmp_path / "s.jsonl", bus)
    rec.start()
    for i in range(5):
        bus.publish(
            RawGazeReady(
                frame_id=i, timestamp=i * 0.033, x=0.5, y=0.5, confidence=1.0, inference_ms=0.0
            )
        )
    rec.stop()
    assert rec.events_written == 5
