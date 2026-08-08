"""Sprint 11 — SessionReport metric calculations."""

from __future__ import annotations

from irisflow.core.events import (
    DwellClick,
    Event,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    PipelineState,
    StateChanged,
)
from irisflow.telemetry.report import (
    build_session_report,
    render_session_report,
)


def _gaze(frame_id: int, ts: float, px: int, py: int, fixation: bool = False) -> GazeUpdated:
    return GazeUpdated(
        frame_id=frame_id,
        timestamp=ts,
        px=px,
        py=py,
        is_fixation=fixation,
        confidence=1.0,
    )


def test_report_counts_frames_and_computes_fps() -> None:
    events: list[Event] = [
        _gaze(0, 0.0, 100, 100),
        _gaze(1, 1.0, 110, 100),
        _gaze(2, 2.0, 120, 100),
    ]
    report = build_session_report(session_id="s1", events=events)
    assert report.frames_ok == 3
    assert report.duration_s == 2.0
    assert report.fps == 1.5


def test_report_zero_duration_reports_zero_fps() -> None:
    events: list[Event] = [_gaze(0, 0.0, 100, 100)]
    report = build_session_report(session_id="s1", events=events)
    assert report.duration_s == 0.0
    assert report.fps == 0.0


def test_jitter_only_counts_consecutive_fixations() -> None:
    events: list[Event] = [
        _gaze(0, 0.0, 100, 100, fixation=True),
        _gaze(1, 0.033, 103, 100, fixation=True),
        _gaze(2, 0.066, 100, 100, fixation=True),
        _gaze(3, 0.099, 500, 500, fixation=False),  # saccade, excluded
        _gaze(4, 0.132, 502, 500, fixation=True),
    ]
    report = build_session_report(session_id="s", events=events)
    # deltas within fixation runs: sqrt(9)=3, sqrt(9)=3
    # RMS = sqrt((9 + 9) / 2) = 3.0
    assert report.jitter_px_rms == 3.0


def test_jitter_is_zero_when_no_fixation_run() -> None:
    events: list[Event] = [
        _gaze(0, 0.0, 0, 0, fixation=True),
        _gaze(1, 0.033, 1, 1, fixation=False),
    ]
    report = build_session_report(session_id="s", events=events)
    assert report.jitter_px_rms == 0.0


def test_click_count_matches_dwell_click_events() -> None:
    events: list[Event] = [
        _gaze(0, 0.0, 100, 100),
        DwellClick(frame_id=1, timestamp=0.1, px=100, py=100, button="left"),
        DwellClick(frame_id=2, timestamp=0.5, px=200, py=200, button="left"),
    ]
    report = build_session_report(session_id="s", events=events)
    assert report.click_count == 2


def test_face_lost_ratio_uses_state_windows() -> None:
    events: list[Event] = [
        StateChanged(previous=PipelineState.IDLE, current=PipelineState.TRACKING, timestamp=0.0),
        StateChanged(previous=PipelineState.TRACKING, current=PipelineState.LOST, timestamp=1.0),
        StateChanged(previous=PipelineState.LOST, current=PipelineState.TRACKING, timestamp=1.5),
        StateChanged(previous=PipelineState.TRACKING, current=PipelineState.IDLE, timestamp=2.0),
    ]
    report = build_session_report(session_id="s", events=events)
    # 0.5s lost out of 2s
    assert report.duration_s == 2.0
    assert report.face_lost_fraction == 0.25


def test_face_events_counted() -> None:
    events: list[Event] = [
        FaceLost(frame_id=0, timestamp=0.0, duration_ms=0.0),
        FaceAcquired(frame_id=1, timestamp=0.5),
        FaceLost(frame_id=2, timestamp=1.0, duration_ms=0.0),
    ]
    report = build_session_report(session_id="s", events=events)
    assert report.face_lost_events == 2
    assert report.face_acquired_events == 1


def test_render_report_includes_key_metrics() -> None:
    report = build_session_report(
        session_id="pretty",
        events=[
            _gaze(0, 0.0, 0, 0, fixation=True),
            _gaze(1, 1.0, 0, 0, fixation=True),
        ],
    )
    text = render_session_report(report)
    assert "session       pretty" in text
    assert "fps" in text
    assert "jitter_rms_px" in text
    assert "clicks" in text
