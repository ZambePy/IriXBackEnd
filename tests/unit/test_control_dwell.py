"""Sprint 10 — dwell-click state machine."""

from __future__ import annotations

import pytest

from irisflow.control.dwell import DwellClicker, DwellParams


def _params(**overrides: object) -> DwellParams:
    kw: dict[str, object] = {"radius_px": 30, "duration_ms": 500, "refractory_ms": 200}
    kw.update(overrides)
    return DwellParams(**kw)  # type: ignore[arg-type]


def test_dwell_params_reject_invalid() -> None:
    with pytest.raises(ValueError, match="radius_px"):
        DwellParams(radius_px=0)
    with pytest.raises(ValueError, match="duration_ms"):
        DwellParams(duration_ms=0)
    with pytest.raises(ValueError, match="refractory_ms"):
        DwellParams(refractory_ms=-1)


def test_first_sample_starts_collecting_at_zero_progress() -> None:
    clicker = DwellClicker(_params())
    decision = clicker.on_sample(100, 100, 0.0)
    assert decision.progress == 0.0
    assert decision.click_at is None
    assert decision.phase == "collecting"


def test_stationary_pointer_reaches_click_at_duration() -> None:
    clicker = DwellClicker(_params(duration_ms=500))
    clicker.on_sample(100, 100, 0.0)
    clicker.on_sample(100, 100, 0.25)
    decision = clicker.on_sample(100, 100, 0.5)
    assert decision.click_at == (100, 100)
    assert decision.progress == 1.0


def test_click_uses_running_centroid_not_last_pixel() -> None:
    clicker = DwellClicker(_params(radius_px=100, duration_ms=200))
    clicker.on_sample(100, 100, 0.0)
    clicker.on_sample(120, 100, 0.1)
    decision = clicker.on_sample(140, 100, 0.2)
    assert decision.click_at is not None
    cx, cy = decision.click_at
    assert cx == 120  # mean of 100, 120, 140
    assert cy == 100


def test_progress_grows_monotonically_while_inside_radius() -> None:
    clicker = DwellClicker(_params(duration_ms=1000))
    d1 = clicker.on_sample(50, 50, 0.0)
    d2 = clicker.on_sample(51, 49, 0.25)
    d3 = clicker.on_sample(49, 51, 0.5)
    assert d1.progress < d2.progress < d3.progress
    assert d1.click_at is None
    assert d2.click_at is None
    assert d3.click_at is None


def test_pointer_leaves_radius_resets_cluster() -> None:
    clicker = DwellClicker(_params(radius_px=20, duration_ms=500))
    clicker.on_sample(100, 100, 0.0)
    clicker.on_sample(105, 100, 0.2)
    # Jump far away → new cluster, progress restarts.
    decision = clicker.on_sample(500, 500, 0.3)
    assert decision.progress == 0.0
    assert decision.click_at is None


def test_refractory_suppresses_click_even_when_pointer_stays() -> None:
    clicker = DwellClicker(_params(duration_ms=100, refractory_ms=1000))
    # Fire a click.
    clicker.on_sample(200, 200, 0.0)
    first = clicker.on_sample(200, 200, 0.15)
    assert first.click_at == (200, 200)
    # Stay inside the radius → refractory.
    d = clicker.on_sample(200, 200, 0.2)
    assert d.phase == "refractory"
    assert d.click_at is None


def test_refractory_clears_after_timeout_and_movement() -> None:
    clicker = DwellClicker(_params(duration_ms=100, refractory_ms=200))
    clicker.on_sample(200, 200, 0.0)
    assert clicker.on_sample(200, 200, 0.15).click_at == (200, 200)
    # Move away — enters refractory-outside.
    d1 = clicker.on_sample(600, 600, 0.16)
    assert d1.phase == "refractory"
    # Wait longer than refractory then move back — dwell should restart.
    d2 = clicker.on_sample(600, 600, 0.5)
    assert d2.phase in ("refractory", "collecting")
    d3 = clicker.on_sample(600, 600, 0.6)
    # Now collecting normally.
    assert d3.phase == "collecting"


def test_reset_clears_all_state() -> None:
    clicker = DwellClicker(_params())
    clicker.on_sample(0, 0, 0.0)
    clicker.on_sample(0, 0, 0.5)  # fires click
    assert clicker.is_refractory
    clicker.reset()
    assert not clicker.is_refractory
    d = clicker.on_sample(0, 0, 1.0)
    assert d.progress == 0.0
    assert d.click_at is None
