"""Sprint 10 — ControlSink wiring between the bus, cursor, dwell and safety."""

from __future__ import annotations

from dataclasses import dataclass, field

from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.safety import RestZone, SafetyGate, Watchdog
from irisflow.core.clock import FakeClock
from irisflow.core.events import (
    DwellClick,
    DwellProgress,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    PipelineState,
    SafetyPaused,
    SafetyResumed,
    StateChanged,
)
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.control_sink import ControlSink


@dataclass
class _RecordingCursor:
    moves: list[tuple[int, int]] = field(default_factory=list)
    clicks: list[str] = field(default_factory=list)
    enable_calls: list[bool] = field(default_factory=list)
    _enabled: bool = False

    def move(self, px: int, py: int) -> None:
        self.moves.append((px, py))

    def click(self, button: str = "left") -> None:
        self.clicks.append(button)

    def enable(self) -> None:
        self._enabled = True
        self.enable_calls.append(True)

    def disable(self) -> None:
        self._enabled = False
        self.enable_calls.append(False)

    @property
    def is_enabled(self) -> bool:
        return self._enabled


def _make_sink(
    *,
    cursor_enabled: bool = True,
    dwell_duration_ms: int = 500,
    dwell_radius_px: int = 40,
    rest_zone: RestZone | None = None,
    watchdog: Watchdog | None = None,
) -> tuple[ControlSink, EventBus, _RecordingCursor, FakeClock]:
    clock = FakeClock()
    bus = EventBus()
    cursor = _RecordingCursor()
    safety = SafetyGate(
        pause_on_face_lost_ms=200,
        rest_zone=rest_zone or RestZone(0, 0, 0, 0),
        clock=clock,
    )
    dwell = DwellClicker(
        DwellParams(
            radius_px=dwell_radius_px,
            duration_ms=dwell_duration_ms,
            refractory_ms=100,
        )
    )
    sink = ControlSink(
        bus=bus,
        cursor=cursor,
        safety=safety,
        dwell=dwell,
        watchdog=watchdog,
        cursor_enabled_on_start=cursor_enabled,
        clock=clock,
    )
    return sink, bus, cursor, clock


def _publish_state(bus: EventBus, previous: PipelineState, current: PipelineState) -> None:
    bus.publish(StateChanged(previous=previous, current=current, timestamp=0.0))


def _publish_gaze(bus: EventBus, *, px: int, py: int, ts: float, frame_id: int = 0) -> None:
    bus.publish(
        GazeUpdated(
            frame_id=frame_id,
            timestamp=ts,
            px=px,
            py=py,
            is_fixation=False,
            confidence=1.0,
        )
    )


def test_start_enables_cursor_when_requested() -> None:
    sink, _bus, cursor, _ = _make_sink(cursor_enabled=True)
    sink.start()
    assert cursor.is_enabled
    sink.stop()
    assert not cursor.is_enabled


def test_start_does_not_enable_cursor_when_disabled_by_default() -> None:
    sink, _bus, cursor, _ = _make_sink(cursor_enabled=False)
    sink.start()
    assert not cursor.is_enabled


def test_gaze_moves_cursor_when_tracking() -> None:
    sink, bus, cursor, _ = _make_sink()
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    _publish_gaze(bus, px=100, py=200, ts=0.0)
    assert cursor.moves == [(100, 200)]


def test_gaze_does_not_move_cursor_when_paused() -> None:
    sink, bus, cursor, _ = _make_sink()
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    _publish_state(bus, PipelineState.TRACKING, PipelineState.PAUSED)
    _publish_gaze(bus, px=1, py=1, ts=0.0)
    assert cursor.moves == []


def test_gaze_does_not_move_cursor_when_lost() -> None:
    sink, bus, cursor, _ = _make_sink()
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    _publish_state(bus, PipelineState.TRACKING, PipelineState.LOST)
    _publish_gaze(bus, px=5, py=5, ts=0.0)
    assert cursor.moves == []


def test_dwell_progress_events_published() -> None:
    sink, bus, cursor, _ = _make_sink(dwell_duration_ms=500)
    events: list[object] = []
    bus.subscribe(DwellProgress, events.append)
    bus.subscribe(DwellClick, events.append)
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    _publish_gaze(bus, px=100, py=100, ts=0.0)
    _publish_gaze(bus, px=100, py=100, ts=0.25)
    _publish_gaze(bus, px=100, py=100, ts=0.5)

    progress_events = [e for e in events if isinstance(e, DwellProgress)]
    click_events = [e for e in events if isinstance(e, DwellClick)]
    assert len(progress_events) >= 2
    assert click_events
    assert click_events[0].px == 100
    assert click_events[0].py == 100
    assert cursor.clicks == ["left"]


def test_no_click_when_state_becomes_paused_between_ticks() -> None:
    sink, bus, cursor, _ = _make_sink(dwell_duration_ms=200)
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    _publish_gaze(bus, px=50, py=50, ts=0.0)
    _publish_state(bus, PipelineState.TRACKING, PipelineState.PAUSED)
    _publish_gaze(bus, px=50, py=50, ts=0.5)
    assert cursor.clicks == []


def test_click_suppressed_when_landing_in_rest_zone() -> None:
    sink, bus, cursor, _ = _make_sink(
        dwell_duration_ms=200,
        dwell_radius_px=100,
        rest_zone=RestZone(0, 0, 400, 400),
    )
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    _publish_gaze(bus, px=50, py=50, ts=0.0)
    _publish_gaze(bus, px=50, py=50, ts=0.25)
    assert cursor.clicks == []
    _publish_gaze(bus, px=500, py=500, ts=0.30)
    _publish_gaze(bus, px=500, py=500, ts=0.55)
    assert cursor.clicks == ["left"]


def test_kill_switch_pauses_and_disables_cursor() -> None:
    sink, bus, cursor, _ = _make_sink()
    events: list[object] = []
    bus.subscribe(SafetyPaused, events.append)
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    sink.on_kill_switch()
    assert not cursor.is_enabled
    assert isinstance(events[-1], SafetyPaused)
    assert events[-1].reason == "kill_switch"


def test_watchdog_stall_pauses_and_disables_cursor() -> None:
    watchdog = Watchdog(timeout_ms=0, on_stall=lambda: None)
    sink, bus, cursor, _ = _make_sink(watchdog=watchdog)
    events: list[object] = []
    bus.subscribe(SafetyPaused, events.append)
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    sink.on_watchdog_stall()
    assert not cursor.is_enabled
    assert events
    assert events[-1].reason == "watchdog"


def test_face_lost_pause_and_face_acquired_resume() -> None:
    sink, bus, cursor, clock = _make_sink()
    events: list[object] = []
    bus.subscribe(SafetyPaused, events.append)
    bus.subscribe(SafetyResumed, events.append)
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    bus.publish(FaceLost(frame_id=1, timestamp=0.0, duration_ms=0.0))
    clock.advance(1.0)
    # A second FaceLost triggers the tick that auto-pauses.
    bus.publish(FaceLost(frame_id=2, timestamp=1.0, duration_ms=0.0))
    assert not cursor.is_enabled
    # Face is back.
    bus.publish(FaceAcquired(frame_id=3, timestamp=1.1))
    assert cursor.is_enabled
    assert any(isinstance(e, SafetyPaused) and e.reason == "face_lost" for e in events)
    assert any(isinstance(e, SafetyResumed) for e in events)


def test_watchdog_kick_is_a_noop_when_watchdog_absent() -> None:
    sink, _bus, _cursor, _ = _make_sink()
    sink.watchdog_kick()  # must not raise


def test_stop_is_idempotent() -> None:
    sink, _bus, _cursor, _ = _make_sink()
    sink.start()
    sink.stop()
    sink.stop()  # no crash


def test_start_is_idempotent() -> None:
    sink, bus, cursor, _ = _make_sink()
    sink.start()
    sink.start()
    _publish_state(bus, PipelineState.IDLE, PipelineState.TRACKING)
    _publish_gaze(bus, px=1, py=1, ts=0.0)
    assert cursor.moves == [(1, 1)]  # not duplicated because subscriptions are unique
