"""Sprint 9 -- One Euro filter.

Tests use synthetic signals -- constant, step, ramp, noise -- and
`FakeClock`-style manual timestamps. The DoD criteria checked here:

* Jitter in fixation <= 15 px RMS (over a noisy stationary signal).
* Latency added during saccades <= 20 ms (measured as ramp lag).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irisflow.filtering.base import SignalSample
from irisflow.filtering.one_euro import OneEuroFilter, _low_pass_alpha


def test_first_sample_passes_through() -> None:
    f = OneEuroFilter()
    s = SignalSample(x=100, y=200, timestamp=0.0)
    assert f.step(s) == s


def test_constant_signal_converges_exactly() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f.step(SignalSample(x=500, y=500, timestamp=0.0))
    for i in range(1, 30):
        out = f.step(SignalSample(x=500, y=500, timestamp=i / 30.0))
    assert out.x == pytest.approx(500.0, abs=1e-9)
    assert out.y == pytest.approx(500.0, abs=1e-9)


def test_reduces_gaussian_noise_jitter_below_15px_rms() -> None:
    """DoD: jitter in fixation <= 15 px RMS."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    rng = np.random.default_rng(0)
    fps = 30.0
    xs: list[float] = []
    ys: list[float] = []
    for i in range(120):
        raw_x = 960 + rng.normal(0.0, 5.0)  # 5 px std raw noise
        raw_y = 540 + rng.normal(0.0, 5.0)
        out = f.step(SignalSample(x=raw_x, y=raw_y, timestamp=i / fps))
        if i > 30:  # skip transient
            xs.append(out.x - 960)
            ys.append(out.y - 540)
    arr = np.asarray(xs) ** 2 + np.asarray(ys) ** 2
    rms = math.sqrt(arr.mean())
    assert rms <= 15.0, f"jitter {rms:.2f} px > 15 px RMS"


def test_lag_on_saccade_ramp_below_20ms() -> None:
    """DoD: latency added by filtering <= 20 ms during saccades."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    fps = 30.0
    # 20 sample dwell then constant-velocity ramp at ~900 px/s
    x0 = 200
    for i in range(20):
        f.step(SignalSample(x=x0, y=540, timestamp=i / fps))
    velocity_px_per_sample = 30.0
    gaps: list[float] = []
    for j in range(40):
        i = 20 + j
        raw_x = x0 + velocity_px_per_sample * j
        out = f.step(SignalSample(x=raw_x, y=540, timestamp=i / fps))
        if j > 10:
            gaps.append(raw_x - out.x)
    median_gap_px = float(np.median(gaps))
    lag_ms = abs(median_gap_px / velocity_px_per_sample) * (1000.0 / fps)
    assert lag_ms <= 20.0, f"ramp lag {lag_ms:.2f} ms > 20 ms"


def test_non_monotonic_timestamps_returns_last_state() -> None:
    f = OneEuroFilter()
    f.step(SignalSample(x=100, y=100, timestamp=1.0))
    out = f.step(SignalSample(x=200, y=200, timestamp=0.5))
    assert out.x == 100
    assert out.y == 100


def test_reset_clears_state() -> None:
    f = OneEuroFilter()
    f.step(SignalSample(x=100, y=100, timestamp=0.0))
    f.step(SignalSample(x=200, y=200, timestamp=0.033))
    f.reset()
    s = SignalSample(x=999, y=999, timestamp=0.0)
    assert f.step(s) == s


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError, match="min_cutoff"):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError, match="beta"):
        OneEuroFilter(beta=-0.1)
    with pytest.raises(ValueError, match="d_cutoff"):
        OneEuroFilter(d_cutoff=0.0)


def test_low_pass_alpha_edge_cases() -> None:
    assert _low_pass_alpha(cutoff=0.0, dt=0.1) == 1.0
    assert _low_pass_alpha(cutoff=1.0, dt=0.0) == 1.0
    a = _low_pass_alpha(cutoff=1.0, dt=1.0 / 30.0)
    assert 0.0 < a < 1.0
