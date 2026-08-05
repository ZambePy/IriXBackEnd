"""Sprint 8 -- per-target sample collector."""

from __future__ import annotations

import pytest

from irisflow.calibration.collector import (
    CollectorState,
    TargetCollector,
    flatten_batches,
)
from irisflow.core.exceptions import CalibrationError
from irisflow.core.types import RawGaze


def _stable_sample(nx: float = 0.5, ny: float = 0.5, jitter: float = 0.001) -> RawGaze:
    return RawGaze(x=nx + jitter, y=ny + jitter, confidence=1.0, inference_ms=1.0)


def test_collector_waits_for_stabilization_then_collects() -> None:
    collector = TargetCollector(
        target_index=0, target_nx=0.5, target_ny=0.5,
        samples_per_point=3, stabilization_ms=500,
        stability_window=2, stability_threshold=0.05,
    )
    collector.mark_shown(timestamp_s=0.0)
    # Before stabilization: samples are seen but not accepted
    collector.push_sample(_stable_sample(), timestamp_s=0.1)
    assert collector.collected == 0
    assert collector.state == CollectorState.WAITING
    # After stabilization window
    for i in range(5):
        collector.push_sample(_stable_sample(), timestamp_s=0.6 + i * 0.05)
    assert collector.collected == 3
    assert collector.state == CollectorState.DONE


def test_collector_face_lost_clears_stability_window() -> None:
    collector = TargetCollector(
        target_index=0, target_nx=0.5, target_ny=0.5,
        samples_per_point=3, stabilization_ms=100,
        stability_window=2, stability_threshold=0.05,
    )
    collector.mark_shown(timestamp_s=0.0)
    # Fill window
    collector.push_sample(_stable_sample(), timestamp_s=0.2)
    # Face lost between samples
    collector.push_sample(None, timestamp_s=0.25)
    # Only one sample -> window not full; nothing accepted
    collector.push_sample(_stable_sample(), timestamp_s=0.3)
    assert collector.collected == 0
    # Now push enough to refill window
    collector.push_sample(_stable_sample(), timestamp_s=0.35)
    assert collector.collected == 1


def test_collector_rejects_unstable_samples() -> None:
    collector = TargetCollector(
        target_index=0, target_nx=0.5, target_ny=0.5,
        samples_per_point=2, stabilization_ms=0,
        stability_window=3, stability_threshold=0.02,
    )
    collector.mark_shown(timestamp_s=0.0)
    # Wildly varying samples => never stable
    collector.push_sample(RawGaze(x=0.1, y=0.1, confidence=1.0, inference_ms=1.0), 0.1)
    collector.push_sample(RawGaze(x=0.9, y=0.9, confidence=1.0, inference_ms=1.0), 0.2)
    collector.push_sample(RawGaze(x=0.1, y=0.1, confidence=1.0, inference_ms=1.0), 0.3)
    assert collector.collected == 0


def test_collector_abort_stops_further_collection() -> None:
    collector = TargetCollector(
        target_index=0, target_nx=0.5, target_ny=0.5,
        samples_per_point=3, stabilization_ms=0,
        stability_window=1, stability_threshold=0.1,
    )
    collector.mark_shown(timestamp_s=0.0)
    collector.abort()
    collector.push_sample(_stable_sample(), timestamp_s=0.1)
    assert collector.state == CollectorState.ABORTED
    assert collector.collected == 0


def test_collector_push_before_mark_shown_raises() -> None:
    collector = TargetCollector(target_index=0, target_nx=0.5, target_ny=0.5)
    with pytest.raises(CalibrationError, match="before mark_shown"):
        collector.push_sample(_stable_sample(), timestamp_s=0.0)


def test_collector_force_done_requires_samples() -> None:
    collector = TargetCollector(target_index=0, target_nx=0.5, target_ny=0.5)
    with pytest.raises(CalibrationError, match="use abort"):
        collector.force_done()


def test_collector_force_done_short_circuits_after_one_sample() -> None:
    collector = TargetCollector(
        target_index=0, target_nx=0.5, target_ny=0.5,
        samples_per_point=100, stabilization_ms=0,
        stability_window=1, stability_threshold=0.1,
    )
    collector.mark_shown(timestamp_s=0.0)
    collector.push_sample(_stable_sample(), timestamp_s=0.1)
    assert collector.collected == 1
    collector.force_done()
    assert collector.is_done


def test_collector_rejects_bad_config() -> None:
    with pytest.raises(CalibrationError, match="samples_per_point"):
        TargetCollector(target_index=0, target_nx=0, target_ny=0, samples_per_point=0)
    with pytest.raises(CalibrationError, match="stabilization_ms"):
        TargetCollector(target_index=0, target_nx=0, target_ny=0, stabilization_ms=-1)
    with pytest.raises(CalibrationError, match="stability_window"):
        TargetCollector(target_index=0, target_nx=0, target_ny=0, stability_window=0)
    with pytest.raises(CalibrationError, match="stability_threshold"):
        TargetCollector(target_index=0, target_nx=0, target_ny=0, stability_threshold=0)


def test_collector_ignores_pushes_after_done() -> None:
    collector = TargetCollector(
        target_index=0, target_nx=0.5, target_ny=0.5,
        samples_per_point=1, stabilization_ms=0,
        stability_window=1, stability_threshold=0.1,
    )
    collector.mark_shown(timestamp_s=0.0)
    collector.push_sample(_stable_sample(), timestamp_s=0.1)
    assert collector.is_done
    # Any further push must not raise or change state.
    collector.push_sample(_stable_sample(), timestamp_s=0.2)
    assert collector.collected == 1


def test_collector_mark_shown_after_collecting_raises() -> None:
    collector = TargetCollector(
        target_index=0, target_nx=0.5, target_ny=0.5,
        samples_per_point=5, stabilization_ms=0,
        stability_window=1, stability_threshold=0.1,
    )
    collector.mark_shown(timestamp_s=0.0)
    collector.push_sample(_stable_sample(), timestamp_s=0.1)  # transitions to COLLECTING
    with pytest.raises(CalibrationError, match="mark_shown"):
        collector.mark_shown(timestamp_s=1.0)


def test_flatten_batches_produces_aligned_arrays() -> None:
    b1 = TargetCollector(target_index=0, target_nx=0.1, target_ny=0.2,
                         samples_per_point=1, stabilization_ms=0,
                         stability_window=1, stability_threshold=0.1)
    b1.mark_shown(0.0)
    b1.push_sample(RawGaze(x=0.11, y=0.21, confidence=1.0, inference_ms=1.0), 0.1)
    b2 = TargetCollector(target_index=1, target_nx=0.9, target_ny=0.8,
                         samples_per_point=1, stabilization_ms=0,
                         stability_window=1, stability_threshold=0.1)
    b2.mark_shown(0.0)
    b2.push_sample(RawGaze(x=0.91, y=0.81, confidence=1.0, inference_ms=1.0), 0.1)
    samples, targets, indices = flatten_batches([b1.batch(), b2.batch()])
    assert len(samples) == len(targets) == len(indices) == 2
    assert targets == [(0.1, 0.2), (0.9, 0.8)]
    assert indices == [0, 1]
