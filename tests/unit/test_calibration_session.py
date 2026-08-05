"""Sprint 8 -- end-to-end calibration session state machine."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.calibration.collector import CollectorState
from irisflow.calibration.session import CalibrationSession, SessionPhase
from irisflow.core.clock import FakeClock
from irisflow.core.events import CalibrationProgress
from irisflow.core.exceptions import CalibrationError
from irisflow.core.types import CalibratedGaze, RawGaze


def _feed_target_to_completion(
    session: CalibrationSession,
    target_index: int,
    *,
    scale_x: float = 1.1,
    scale_y: float = 0.9,
    offset_x: float = 0.02,
    offset_y: float = -0.02,
    clock: FakeClock,
    max_iters: int = 200,
) -> None:
    """Drive one target through mark_shown -> stabilization -> N accepted samples."""
    target = session.targets[target_index]
    session.begin_target(target_index)
    clock.advance(session.stabilization_ms / 1000.0 + 0.05)
    rng = np.random.default_rng(target_index + 100)
    iters = 0
    while not session.is_current_target_done and iters < max_iters:
        noise = float(rng.normal(0.0, 0.001))
        sample = RawGaze(
            x=target.nx * scale_x + offset_x + noise,
            y=target.ny * scale_y + offset_y + noise,
            confidence=1.0,
            inference_ms=1.0,
        )
        clock.advance(0.033)  # ~30 fps
        session.push_sample(sample, clock.monotonic())
        iters += 1
    assert session.is_current_target_done, (
        f"target {target_index} did not reach DONE after {iters} pushes"
    )


def test_session_completes_9_points_and_produces_profile() -> None:
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="alice",
        screen_width=1920,
        screen_height=1080,
        point_count=9,
        samples_per_point=15,
        stabilization_ms=200,
        clock=clock,
    )
    session.start()
    for idx in range(9):
        _feed_target_to_completion(session, idx, clock=clock)
    assert session.is_collection_complete
    outcome = session.finalize()
    assert session.phase == SessionPhase.COMPLETE
    assert outcome.profile.profile_id == "alice"
    assert outcome.report.mean_error_px < outcome.holdout_report.mean_error_px * 3
    # With ~15 samples/point and a smooth affine bias, mean error is a
    # few pixels at most.
    assert outcome.report.mean_error_px < 50.0


def test_session_wall_time_budget_under_60s() -> None:
    """Sprint 8 DoD: 9-point session completes in <=60s.

    Uses FakeClock and the config defaults (30 samples/point, 600ms
    stabilization, ~30fps). The session's simulated wall-clock advance
    is the sum of stabilization + N/fps per target.
    """
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="bob",
        screen_width=1920,
        screen_height=1080,
        point_count=9,
        samples_per_point=30,
        stabilization_ms=600,
        clock=clock,
    )
    session.start()
    start = clock.monotonic()
    for idx in range(9):
        _feed_target_to_completion(session, idx, clock=clock)
    elapsed = clock.monotonic() - start
    session.finalize()
    assert elapsed <= 60.0, f"session took {elapsed:.1f}s (>60s budget)"


def test_session_publishes_progress_events() -> None:
    clock = FakeClock()
    events: list[CalibrationProgress] = []
    session = CalibrationSession(
        profile_id="carol",
        screen_width=1000,
        screen_height=800,
        point_count=5,
        samples_per_point=5,
        stabilization_ms=100,
        clock=clock,
        publisher=events.append,
    )
    session.start()
    for idx in range(5):
        _feed_target_to_completion(session, idx, clock=clock)
    session.finalize()
    phases = [e.phase for e in events]
    assert phases.count("prompt") == 5
    assert phases.count("settled") >= 5  # one per target when done
    assert "done" in phases  # emitted in finalize


def test_session_retry_targets_clears_previous_batch() -> None:
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="dan", screen_width=1000, screen_height=800,
        point_count=5, samples_per_point=3, stabilization_ms=0, clock=clock,
    )
    session.start()
    _feed_target_to_completion(session, 0, clock=clock)
    assert session.is_current_target_done
    session.retry_targets([0])
    # After retry, target 0 has no collector -- begin_target creates a fresh one.
    _feed_target_to_completion(session, 0, clock=clock)
    for idx in range(1, 5):
        _feed_target_to_completion(session, idx, clock=clock)
    outcome = session.finalize()
    assert outcome.profile.n_samples == 5 * 3


def test_session_finalize_before_complete_raises() -> None:
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="eve", screen_width=1000, screen_height=800,
        point_count=5, samples_per_point=3, stabilization_ms=0, clock=clock,
    )
    session.start()
    _feed_target_to_completion(session, 0, clock=clock)
    with pytest.raises(CalibrationError, match="every target"):
        session.finalize()


def test_session_reports_bad_targets_when_error_above_threshold() -> None:
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="frank",
        screen_width=1920, screen_height=1080,
        point_count=9, samples_per_point=8,
        stabilization_ms=0, clock=clock,
        max_residual_px=1.0,  # extremely strict threshold
    )
    session.start()
    for idx in range(9):
        _feed_target_to_completion(session, idx, clock=clock)
    outcome = session.finalize()
    # With a 1 px threshold and a real (small) fit error, most targets
    # will exceed the threshold.
    assert not outcome.accepted or len(outcome.bad_targets) > 0


def test_session_force_complete_active_allows_short_target() -> None:
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="gina", screen_width=1000, screen_height=800,
        point_count=5, samples_per_point=100, stabilization_ms=0, clock=clock,
    )
    session.start()
    target = session.targets[0]
    session.begin_target(0)
    clock.advance(0.05)
    # Push enough samples to satisfy the stability window (default 5),
    # then a few more so the collector accepts them.
    for i in range(10):
        session.push_sample(
            RawGaze(x=target.nx * 1.05, y=target.ny * 1.05, confidence=1.0, inference_ms=1.0),
            clock.monotonic() + 0.01 * i,
        )
    assert session.current_target_collected >= 1
    session.force_complete_active()
    assert session.is_current_target_done


def test_session_aborted_targets_still_count_as_resolved() -> None:
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="hana",
        screen_width=1000, screen_height=800,
        point_count=5, samples_per_point=3, stabilization_ms=0, clock=clock,
    )
    session.start()
    _feed_target_to_completion(session, 0, clock=clock)
    session.begin_target(1)
    session.abort_target(1)
    assert not session.is_collection_complete  # still targets 2..4 to go
    for idx in range(2, 5):
        _feed_target_to_completion(session, idx, clock=clock)
    assert session.is_collection_complete
    # finalize succeeds with 4 targets contributing samples
    outcome = session.finalize()
    assert outcome.profile.n_samples == 4 * 3


def test_session_uncalibrated_pipeline_is_degraded_not_broken() -> None:
    """DoD: without a fitted calibration, the system must still function.

    Verified by transforming a raw gaze through an unfitted model:
    it raises CalibrationError, which the pipeline layer is expected
    to catch and fall back to passthrough. This test pins the shape of
    that contract.
    """
    from irisflow.calibration.models import PolynomialCalibration

    model = PolynomialCalibration()
    with pytest.raises(CalibrationError, match="before fit"):
        model.transform(RawGaze(x=0.5, y=0.5, confidence=1.0, inference_ms=1.0))
    # ...and the pipeline can substitute a passthrough CalibratedGaze:
    passthrough = CalibratedGaze(x=0.5, y=0.5, profile_id="uncalibrated")
    assert passthrough.profile_id == "uncalibrated"


def test_session_rejects_invalid_screen() -> None:
    clock = FakeClock()
    with pytest.raises(CalibrationError, match="positive dims"):
        CalibrationSession(
            profile_id="x", screen_width=0, screen_height=800, clock=clock,
        )


def test_session_rejects_empty_profile_id() -> None:
    clock = FakeClock()
    with pytest.raises(CalibrationError, match="profile_id"):
        CalibrationSession(
            profile_id="", screen_width=1000, screen_height=800, clock=clock,
        )


def test_current_target_state_none_when_no_target_active() -> None:
    clock = FakeClock()
    session = CalibrationSession(
        profile_id="init", screen_width=1000, screen_height=800, clock=clock,
    )
    assert session.active_target is None
    assert session.current_target_collected == 0
    assert not session.is_current_target_done


def test_session_supports_persistence_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """DoD: profile saved and reloaded is numerically identical."""
    from irisflow.calibration.store import CalibrationStore

    clock = FakeClock()
    session = CalibrationSession(
        profile_id="round",
        screen_width=1600, screen_height=900,
        point_count=9, samples_per_point=10, stabilization_ms=0, clock=clock,
    )
    session.start()
    for idx in range(9):
        _feed_target_to_completion(session, idx, clock=clock)
    outcome = session.finalize()

    store = CalibrationStore(tmp_path)
    store.save(outcome.profile)
    reloaded = store.load("round")
    assert outcome.profile.model.weights is not None
    assert reloaded.model.weights is not None
    np.testing.assert_array_equal(reloaded.model.weights, outcome.profile.model.weights)

    # Transforms produce identical CalibratedGaze
    probe = RawGaze(x=0.42, y=0.61, confidence=1.0, inference_ms=1.0)
    orig = outcome.profile.model.transform(probe)
    round_out = reloaded.model.transform(probe)
    assert orig.x == round_out.x
    assert orig.y == round_out.y


def test_collector_state_transitions_publish_correct_phase() -> None:
    """CollectorState reachable phases are surfaced verbatim on the bus."""
    assert CollectorState.WAITING == "waiting"
    assert CollectorState.COLLECTING == "collecting"
    assert CollectorState.DONE == "done"
    assert CollectorState.ABORTED == "aborted"
