"""Sprint 9 -- FixationClassifier."""

from __future__ import annotations

import pytest

from irisflow.filtering.base import SignalSample
from irisflow.filtering.fixation import FixationClassifier


def test_first_sample_is_fixation_with_zero_velocity() -> None:
    fc = FixationClassifier(velocity_threshold_px_per_s=40.0)
    d = fc.classify(SignalSample(x=100, y=100, timestamp=0.0))
    assert d.velocity_px_per_s == 0.0
    assert d.is_fixation is True


def test_still_signal_classified_as_fixation() -> None:
    fc = FixationClassifier(velocity_threshold_px_per_s=40.0)
    fc.classify(SignalSample(x=100, y=100, timestamp=0.0))
    d = fc.classify(SignalSample(x=100, y=100, timestamp=1.0))
    assert d.is_fixation is True
    assert d.velocity_px_per_s == pytest.approx(0.0)


def test_fast_motion_classified_as_saccade() -> None:
    fc = FixationClassifier(velocity_threshold_px_per_s=40.0)
    fc.classify(SignalSample(x=100, y=100, timestamp=0.0))
    d = fc.classify(SignalSample(x=200, y=100, timestamp=0.1))  # 1000 px/s
    assert d.is_fixation is False
    assert d.velocity_px_per_s == pytest.approx(1000.0)


def test_slow_drift_classified_as_fixation() -> None:
    fc = FixationClassifier(velocity_threshold_px_per_s=40.0)
    fc.classify(SignalSample(x=100, y=100, timestamp=0.0))
    d = fc.classify(SignalSample(x=101, y=100, timestamp=0.1))  # 10 px/s
    assert d.is_fixation is True


def test_non_monotonic_timestamp_defaults_to_fixation() -> None:
    fc = FixationClassifier(velocity_threshold_px_per_s=40.0)
    fc.classify(SignalSample(x=100, y=100, timestamp=1.0))
    d = fc.classify(SignalSample(x=200, y=200, timestamp=0.5))
    assert d.is_fixation is True
    assert d.velocity_px_per_s == 0.0


def test_reset_clears_state() -> None:
    fc = FixationClassifier(velocity_threshold_px_per_s=40.0)
    fc.classify(SignalSample(x=100, y=100, timestamp=0.0))
    fc.reset()
    d = fc.classify(SignalSample(x=999, y=999, timestamp=1.0))
    assert d.is_fixation is True
    assert d.velocity_px_per_s == 0.0


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="velocity_threshold"):
        FixationClassifier(velocity_threshold_px_per_s=0.0)
