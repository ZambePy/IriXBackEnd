"""Sprint 9 -- OutlierRejector: freeze cursor on unphysical jumps."""

from __future__ import annotations

import pytest

from irisflow.filtering.base import SignalSample
from irisflow.filtering.outlier import OutlierRejector


def test_first_sample_passes_through() -> None:
    rej = OutlierRejector(max_velocity_px_per_s=1000.0)
    s = SignalSample(x=100, y=100, timestamp=0.0)
    assert rej.step(s) == s


def test_slow_motion_passes_through() -> None:
    rej = OutlierRejector(max_velocity_px_per_s=1000.0)
    rej.step(SignalSample(x=0, y=0, timestamp=0.0))
    # 10 px / 1 s = 10 px/s -- well under threshold
    out = rej.step(SignalSample(x=10, y=0, timestamp=1.0))
    assert out.x == 10


def test_fast_jump_freezes_on_last() -> None:
    rej = OutlierRejector(max_velocity_px_per_s=1000.0)
    rej.step(SignalSample(x=0, y=0, timestamp=0.0))
    # 5000 px in 0.001 s = 5,000,000 px/s -- way over threshold
    out = rej.step(SignalSample(x=5000, y=0, timestamp=0.001))
    assert out.x == 0
    assert out.y == 0


def test_backwards_time_returns_last() -> None:
    rej = OutlierRejector(max_velocity_px_per_s=1000.0)
    rej.step(SignalSample(x=100, y=100, timestamp=1.0))
    out = rej.step(SignalSample(x=200, y=200, timestamp=0.5))
    assert out.x == 100
    assert out.y == 100


def test_reset_clears_state() -> None:
    rej = OutlierRejector(max_velocity_px_per_s=1000.0)
    rej.step(SignalSample(x=100, y=100, timestamp=0.0))
    rej.reset()
    # Next sample is treated as the first again -- unbounded velocity ok.
    out = rej.step(SignalSample(x=9999, y=9999, timestamp=0.001))
    assert out.x == 9999


def test_zero_or_negative_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="max_velocity"):
        OutlierRejector(max_velocity_px_per_s=0)
