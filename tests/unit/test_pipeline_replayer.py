"""Sprint 11 — SessionReplayer produces deterministic output."""

from __future__ import annotations

from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.safety import RestZone, SafetyGate
from irisflow.core.clock import FakeClock
from irisflow.core.events import (
    Event,
    FaceAcquired,
    FaceLost,
    PipelineState,
    RawGazeReady,
    StateChanged,
)
from irisflow.filtering.chain import (
    ChainConfig,
    FixationParams,
    OneEuroParams,
    OutlierParams,
    build_filter_chain,
)
from irisflow.mapping.screen import ScreenMapper as ScreenPixelMapper
from irisflow.mapping.screen_info import ScreenInfo
from irisflow.pipeline.replayer import SessionReplayer


def _build_replayer(cursor_enabled: bool = True) -> SessionReplayer:
    clock = FakeClock()
    return SessionReplayer(
        calibration_model=None,
        mapper=ScreenPixelMapper(
            screen=ScreenInfo(width_px=1000, height_px=800),
            clamp_margin_px=0,
        ),
        filter_chain=build_filter_chain(
            ChainConfig(
                chain=("outlier", "one_euro", "fixation"),
                outlier=OutlierParams(max_velocity_px_per_s=100_000.0),
                one_euro=OneEuroParams(min_cutoff=10.0, beta=0.5, d_cutoff=10.0),
                fixation=FixationParams(velocity_threshold_px_per_s=50.0),
            )
        ),
        safety=SafetyGate(
            pause_on_face_lost_ms=1000,
            rest_zone=RestZone(0, 0, 0, 0),
            clock=clock,
        ),
        dwell=DwellClicker(
            DwellParams(radius_px=200, duration_ms=50, refractory_ms=20)
        ),
        cursor_enabled=cursor_enabled,
        clock=clock,
    )


def _raw(frame_id: int, ts: float, x: float = 0.5, y: float = 0.5) -> RawGazeReady:
    return RawGazeReady(
        frame_id=frame_id,
        timestamp=ts,
        x=x,
        y=y,
        confidence=1.0,
        inference_ms=0.0,
    )


def test_same_input_produces_same_output_twice() -> None:
    events: list[Event] = [
        StateChanged(PipelineState.IDLE, PipelineState.TRACKING, 0.0),
        _raw(0, 0.0),
        _raw(1, 0.033),
        _raw(2, 0.066),
        _raw(3, 0.099),
        _raw(4, 0.132),
    ]
    result_a = _build_replayer().replay(events)
    result_b = _build_replayer().replay(events)
    assert result_a.gaze_updates == result_b.gaze_updates
    assert result_a.dwell_progress == result_b.dwell_progress
    assert result_a.dwell_clicks == result_b.dwell_clicks


def test_replay_produces_dwell_click_after_stationary_gaze() -> None:
    events: list[Event] = [
        StateChanged(PipelineState.IDLE, PipelineState.TRACKING, 0.0),
    ]
    for i in range(6):
        events.append(_raw(i, i * 0.033, x=0.5, y=0.5))
    result = _build_replayer().replay(events)
    assert result.gaze_updates, "expected at least one GazeUpdated"
    assert result.dwell_clicks, "stationary gaze should trigger a dwell click"


def test_replay_skips_dwell_when_cursor_disabled() -> None:
    events: list[Event] = [
        StateChanged(PipelineState.IDLE, PipelineState.TRACKING, 0.0),
        _raw(0, 0.0),
        _raw(1, 0.05),
        _raw(2, 0.10),
    ]
    result = _build_replayer(cursor_enabled=False).replay(events)
    assert result.gaze_updates
    assert result.dwell_progress == []
    assert result.dwell_clicks == []


def test_replay_state_changed_freezes_gaze_updates_during_lost() -> None:
    events: list[Event] = [
        StateChanged(PipelineState.IDLE, PipelineState.TRACKING, 0.0),
        _raw(0, 0.0),
        StateChanged(PipelineState.TRACKING, PipelineState.LOST, 0.05),
        _raw(1, 0.10),  # replayed anyway — mapper is deterministic on any input
        StateChanged(PipelineState.LOST, PipelineState.TRACKING, 0.20),
        _raw(2, 0.25),
    ]
    result = _build_replayer().replay(events)
    # Every raw gaze produces a GazeUpdated (mapping/filtering are stateless
    # w.r.t. pipeline state); only the dwell + cursor gate reacts to LOST.
    assert len(result.gaze_updates) == 3


def test_face_lost_pause_and_resume_events_when_cursor_enabled() -> None:
    events: list[Event] = [
        StateChanged(PipelineState.IDLE, PipelineState.TRACKING, 0.0),
        FaceLost(frame_id=1, timestamp=0.0, duration_ms=0.0),
    ]
    # Enough gap between FaceLost + FaceAcquired to trip pause_on_face_lost_ms=1000
    events.append(FaceLost(frame_id=2, timestamp=1.5, duration_ms=0.0))
    events.append(FaceAcquired(frame_id=3, timestamp=1.6))
    result = _build_replayer().replay(events)
    assert result.safety_paused, "expected a face_lost SafetyPaused"
    assert result.safety_resumed, "expected a SafetyResumed after face acquired"
