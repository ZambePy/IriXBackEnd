"""Sprint 7 — MetricsRecorder: stage timing, counters, snapshot."""

from __future__ import annotations

import pytest

from irisflow.core.clock import FakeClock
from irisflow.telemetry.metrics import MetricsRecorder


def test_time_records_stage_duration_in_ms() -> None:
    clock = FakeClock()
    rec = MetricsRecorder(_clock=clock)
    with rec.time("stage_a"):
        clock.advance(0.010)  # 10 ms
    snap = rec.snapshot()
    assert "stage_a" in snap.stages
    assert snap.stages["stage_a"].count == 1
    assert snap.stages["stage_a"].p50_ms == pytest.approx(10.0)


def test_snapshot_computes_percentiles_over_many_samples() -> None:
    clock = FakeClock()
    rec = MetricsRecorder(_clock=clock)
    for delta_s in (0.005, 0.010, 0.020, 0.100):
        with rec.time("infer"):
            clock.advance(delta_s)
    snap = rec.snapshot()
    stats = snap.stages["infer"]
    assert stats.count == 4
    assert stats.max_ms == pytest.approx(100.0)
    assert stats.mean_ms == pytest.approx((5 + 10 + 20 + 100) / 4)


def test_counters_and_fps() -> None:
    clock = FakeClock()
    rec = MetricsRecorder(_clock=clock)
    for _ in range(30):
        rec.increment("frames_ok")
    clock.advance(1.0)
    snap = rec.snapshot()
    assert snap.frames_ok == 30
    assert snap.fps == pytest.approx(30.0, rel=1e-3)


def test_counters_track_drops_and_face_lost() -> None:
    rec = MetricsRecorder()
    rec.increment("frames_dropped", 3)
    rec.increment("frames_face_lost", 2)
    snap = rec.snapshot()
    assert snap.frames_dropped == 3
    assert snap.frames_face_lost == 2


def test_reset_clears_state_and_restarts_window() -> None:
    clock = FakeClock()
    rec = MetricsRecorder(_clock=clock)
    with rec.time("s"):
        clock.advance(0.005)
    rec.increment("frames_ok")
    rec.reset()
    snap = rec.snapshot()
    assert snap.frames_ok == 0
    assert snap.stages == {}


def test_snapshot_empty_stages_when_never_timed() -> None:
    snap = MetricsRecorder().snapshot()
    assert snap.stages == {}
    assert snap.fps == pytest.approx(0.0)
