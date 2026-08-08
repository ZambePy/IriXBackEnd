"""Sprint 10 — SafetyGate, RestZone and Watchdog."""

from __future__ import annotations

import threading
import time

import pytest

from irisflow.control.safety import RestZone, SafetyGate, Watchdog
from irisflow.core.clock import FakeClock
from irisflow.core.events import PipelineState


# ---------------------------------------------------------------------------
# RestZone
# ---------------------------------------------------------------------------
def test_rest_zone_zero_area_is_disabled() -> None:
    assert not RestZone(0, 0, 0, 0).is_enabled()
    assert not RestZone(10, 10, 0, 100).is_enabled()


def test_rest_zone_contains_only_when_enabled() -> None:
    z = RestZone(10, 10, 100, 100)
    assert z.is_enabled()
    assert z.contains(20, 20)
    assert z.contains(10, 10)  # top-left inclusive
    assert not z.contains(110, 20)  # right edge exclusive
    assert not z.contains(20, 110)


# ---------------------------------------------------------------------------
# SafetyGate
# ---------------------------------------------------------------------------
def _gate(**over: object) -> SafetyGate:
    kw: dict[str, object] = {
        "pause_on_face_lost_ms": 500,
        "rest_zone": RestZone(0, 0, 0, 0),
    }
    kw.update(over)
    return SafetyGate(**kw)  # type: ignore[arg-type]


def test_can_move_only_when_tracking_and_not_paused() -> None:
    gate = _gate()
    assert not gate.can_move()  # IDLE by default
    gate.on_pipeline_state(PipelineState.TRACKING)
    assert gate.can_move()
    gate.on_pipeline_state(PipelineState.LOST)
    assert not gate.can_move()
    gate.on_pipeline_state(PipelineState.TRACKING)
    gate.trigger_pause("manual")
    assert not gate.can_move()
    gate.resume()
    assert gate.can_move()


def test_face_lost_pause_kicks_in_after_timeout() -> None:
    clock = FakeClock()
    gate = _gate(pause_on_face_lost_ms=100, clock=clock)
    gate.on_pipeline_state(PipelineState.TRACKING)
    gate.on_face_lost()
    assert not gate.is_paused  # not enough time yet
    clock.advance(0.05)
    gate.tick()
    assert not gate.is_paused
    clock.advance(0.1)
    gate.tick()
    assert gate.is_paused
    assert gate.pause_reason == "face_lost"


def test_face_acquired_clears_face_lost_pause() -> None:
    clock = FakeClock()
    gate = _gate(pause_on_face_lost_ms=50, clock=clock)
    gate.on_pipeline_state(PipelineState.TRACKING)
    gate.on_face_lost()
    clock.advance(0.1)
    gate.tick()
    assert gate.is_paused
    gate.on_face_acquired()
    assert not gate.is_paused


def test_face_acquired_does_not_clear_kill_switch_pause() -> None:
    gate = _gate()
    gate.on_pipeline_state(PipelineState.TRACKING)
    gate.trigger_pause("kill_switch")
    gate.on_face_acquired()
    assert gate.is_paused
    assert gate.pause_reason == "kill_switch"


def test_can_click_false_when_in_rest_zone() -> None:
    gate = _gate(rest_zone=RestZone(0, 0, 200, 200))
    gate.on_pipeline_state(PipelineState.TRACKING)
    assert not gate.can_click(50, 50)
    assert gate.can_click(500, 500)


def test_can_click_false_when_paused_or_not_tracking() -> None:
    gate = _gate()
    gate.on_pipeline_state(PipelineState.PAUSED)
    assert not gate.can_click(0, 0)
    gate.on_pipeline_state(PipelineState.TRACKING)
    gate.trigger_pause("kill_switch")
    assert not gate.can_click(0, 0)


def test_snapshot_reflects_current_state() -> None:
    clock = FakeClock()
    gate = _gate(pause_on_face_lost_ms=1000, clock=clock)
    gate.on_pipeline_state(PipelineState.TRACKING)
    gate.on_face_lost()
    snap = gate.snapshot()
    assert snap.pipeline_state == PipelineState.TRACKING
    assert snap.face_lost_since_s == pytest.approx(0.0)
    assert not snap.paused


def test_pause_on_face_lost_zero_disables_auto_pause() -> None:
    clock = FakeClock()
    gate = _gate(pause_on_face_lost_ms=0, clock=clock)
    gate.on_pipeline_state(PipelineState.TRACKING)
    gate.on_face_lost()
    clock.advance(60.0)
    gate.tick()
    assert not gate.is_paused


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------
def test_watchdog_check_returns_false_before_timeout() -> None:
    clock = FakeClock()
    fired = {"n": 0}

    def _on_stall() -> None:
        fired["n"] += 1

    dog = Watchdog(timeout_ms=100, on_stall=_on_stall, clock=clock)
    assert not dog.check()
    clock.advance(0.05)
    assert not dog.check()
    assert fired["n"] == 0


def test_watchdog_fires_after_timeout_and_can_be_kicked() -> None:
    clock = FakeClock()
    fired = {"n": 0}

    def _on_stall() -> None:
        fired["n"] += 1

    dog = Watchdog(timeout_ms=100, on_stall=_on_stall, clock=clock)
    clock.advance(0.2)
    assert dog.check()
    assert fired["n"] == 1
    # kick resets the timer.
    dog.kick()
    clock.advance(0.05)
    assert not dog.check()
    assert fired["n"] == 1


def test_watchdog_zero_timeout_is_disabled() -> None:
    fired = {"n": 0}
    dog = Watchdog(timeout_ms=0, on_stall=lambda: fired.__setitem__("n", 1))
    assert not dog.check()
    assert fired["n"] == 0


def test_watchdog_thread_can_start_and_stop() -> None:
    # Real thread here — verifies start/stop lifecycle without needing to
    # observe timing (the FakeClock tests cover behaviour).
    fired = threading.Event()
    dog = Watchdog(timeout_ms=50, on_stall=fired.set)
    dog.start(poll_interval_s=0.02)
    fired.wait(timeout=1.0)
    dog.stop()
    assert fired.is_set()
    # A second stop is a no-op.
    dog.stop()


def test_watchdog_start_does_nothing_when_disabled() -> None:
    dog = Watchdog(timeout_ms=0, on_stall=lambda: None)
    dog.start()  # must not spawn a thread
    time.sleep(0.05)
    dog.stop()
