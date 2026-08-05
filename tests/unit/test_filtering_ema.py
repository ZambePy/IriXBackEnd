"""Sprint 9 -- EMA baseline smoother."""

from __future__ import annotations

import pytest

from irisflow.filtering.base import SignalSample
from irisflow.filtering.ema import EmaFilter


def test_first_sample_passes_through() -> None:
    ema = EmaFilter(alpha=0.5)
    s = SignalSample(x=100, y=200, timestamp=0.0)
    assert ema.step(s) == s


def test_ema_geometric_convergence_on_step() -> None:
    ema = EmaFilter(alpha=0.5)
    ema.step(SignalSample(x=0, y=0, timestamp=0.0))
    out1 = ema.step(SignalSample(x=100, y=0, timestamp=0.01))
    out2 = ema.step(SignalSample(x=100, y=0, timestamp=0.02))
    out3 = ema.step(SignalSample(x=100, y=0, timestamp=0.03))
    # 0 -> 50 -> 75 -> 87.5
    assert out1.x == pytest.approx(50.0)
    assert out2.x == pytest.approx(75.0)
    assert out3.x == pytest.approx(87.5)


def test_alpha_one_is_passthrough() -> None:
    ema = EmaFilter(alpha=1.0)
    ema.step(SignalSample(x=0, y=0, timestamp=0.0))
    out = ema.step(SignalSample(x=42, y=13, timestamp=1.0))
    assert out.x == 42
    assert out.y == 13


def test_reset_restores_first_sample_behavior() -> None:
    ema = EmaFilter(alpha=0.4)
    ema.step(SignalSample(x=0, y=0, timestamp=0.0))
    ema.reset()
    out = ema.step(SignalSample(x=100, y=100, timestamp=0.0))
    assert out.x == 100


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError, match="alpha"):
        EmaFilter(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        EmaFilter(alpha=1.5)
