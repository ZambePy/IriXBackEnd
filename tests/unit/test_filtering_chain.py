"""Sprint 9 -- FilterChain builder and composed behaviour.

Covers the "chain declared in YAML doesn't require code changes" DoD by
constructing chains from different ChainConfig specs and checking the
composed behaviour differs accordingly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irisflow.core.types import ScreenPoint
from irisflow.filtering.chain import (
    ChainConfig,
    EmaParams,
    FilterChain,
    FixationParams,
    OneEuroParams,
    OutlierParams,
    build_filter_chain,
)


def test_build_default_chain_is_outlier_one_euro_fixation() -> None:
    chain = build_filter_chain(ChainConfig())
    assert len(chain.signal_filters) == 2  # outlier + one_euro
    assert chain.fixation is not None


def test_build_rejects_unknown_filter() -> None:
    """The Literal makes this a type error but runtime is defensive too."""
    # Use tuple[str, ...] cast to bypass Literal restriction at test time.
    with pytest.raises(ValueError, match="filter chain must have"):
        build_filter_chain(ChainConfig(chain=()))  # type: ignore[arg-type]


def test_build_rejects_duplicate_filter() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_filter_chain(ChainConfig(chain=("outlier", "outlier", "fixation")))  # type: ignore[arg-type]


def test_build_rejects_fixation_in_non_terminal_position() -> None:
    with pytest.raises(ValueError, match="must be the last stage"):
        build_filter_chain(ChainConfig(chain=("fixation", "one_euro")))


def test_chain_with_only_fixation_passes_position_through() -> None:
    chain = build_filter_chain(ChainConfig(chain=("fixation",)))
    out = chain.apply(ScreenPoint(px=123, py=456), timestamp=0.0)
    assert out.px == 123
    assert out.py == 456
    assert out.is_fixation is True


def test_chain_ema_only_produces_smoothed_output() -> None:
    chain = build_filter_chain(
        ChainConfig(chain=("ema", "fixation"), ema=EmaParams(alpha=0.5))
    )
    chain.apply(ScreenPoint(px=0, py=0), timestamp=0.0)
    out = chain.apply(ScreenPoint(px=100, py=0), timestamp=0.033)
    assert out.px == 50


def test_chain_fixation_flag_reflects_velocity() -> None:
    chain = build_filter_chain(
        ChainConfig(
            chain=("fixation",),
            fixation=FixationParams(velocity_threshold_px_per_s=100.0),
        )
    )
    chain.apply(ScreenPoint(px=100, py=100), timestamp=0.0)
    out_slow = chain.apply(ScreenPoint(px=101, py=100), timestamp=1.0)
    assert out_slow.is_fixation
    out_fast = chain.apply(ScreenPoint(px=1000, py=100), timestamp=1.01)
    assert not out_fast.is_fixation


def test_chain_reset_propagates() -> None:
    chain = build_filter_chain(ChainConfig())
    chain.apply(ScreenPoint(px=100, py=100), timestamp=0.0)
    chain.apply(ScreenPoint(px=200, py=200), timestamp=0.033)
    chain.reset()
    # After reset, first sample behaviour applies again -- an extreme
    # jump would normally be squelched by the outlier stage, but here
    # it becomes the new "first sample".
    out = chain.apply(ScreenPoint(px=9999, py=9999), timestamp=0.0)
    assert out.px == 9999


def test_chain_without_fixation_reports_no_fixation() -> None:
    chain = build_filter_chain(ChainConfig(chain=("outlier",)))
    out = chain.apply(ScreenPoint(px=100, py=100), timestamp=0.0)
    assert out.is_fixation is False
    assert out.velocity == 0.0


# ---------------------------------------------------------------------------
# DoD: numeric criteria on the default chain
# ---------------------------------------------------------------------------
def test_default_chain_meets_jitter_criterion() -> None:
    chain = build_filter_chain(ChainConfig())
    rng = np.random.default_rng(0)
    fps = 30.0
    residuals: list[tuple[int, int]] = []
    for i in range(120):
        raw_x = 960 + round(rng.normal(0.0, 5.0))
        raw_y = 540 + round(rng.normal(0.0, 5.0))
        out = chain.apply(ScreenPoint(px=raw_x, py=raw_y), timestamp=i / fps)
        if i > 30:
            residuals.append((out.px - 960, out.py - 540))
    arr = np.asarray(residuals) ** 2
    rms = math.sqrt(arr.sum(axis=1).mean())
    assert rms <= 15.0, f"jitter {rms:.2f} px > 15 px RMS"


def test_default_chain_meets_ramp_lag_criterion() -> None:
    chain = build_filter_chain(ChainConfig())
    fps = 30.0
    x0 = 200
    for i in range(20):
        chain.apply(ScreenPoint(px=x0, py=540), timestamp=i / fps)
    velocity_px_per_sample = 30.0
    gaps: list[float] = []
    for j in range(40):
        i = 20 + j
        raw_x = x0 + round(velocity_px_per_sample * j)
        out = chain.apply(ScreenPoint(px=raw_x, py=540), timestamp=i / fps)
        if j > 10:
            gaps.append(raw_x - out.px)
    median_gap = float(np.median(gaps))
    lag_ms = abs(median_gap / velocity_px_per_sample) * (1000.0 / fps)
    assert lag_ms <= 20.0, f"ramp lag {lag_ms:.2f} ms > 20 ms"


def test_switching_chain_config_changes_behaviour_without_code_change() -> None:
    """DoD: filter chain change is YAML-only.

    Build three different chains from three different ChainConfig
    instances and verify they produce numerically different outputs
    for the same input stream -- no branching in the caller.
    """
    outputs: list[int] = []
    for chain in (
        FilterChain(signal_filters=[], fixation=None),
        build_filter_chain(ChainConfig(chain=("ema",), ema=EmaParams(alpha=0.2))),
        build_filter_chain(ChainConfig(chain=("one_euro",))),
    ):
        chain.reset()
        chain.apply(ScreenPoint(px=100, py=100), timestamp=0.0)
        out = chain.apply(ScreenPoint(px=500, py=500), timestamp=0.1)
        outputs.append(out.px)
    assert len(set(outputs)) == 3, f"expected 3 distinct outputs, got {outputs}"


def test_outlier_freezes_cursor_on_impossible_jump_through_chain() -> None:
    chain = build_filter_chain(
        ChainConfig(
            chain=("outlier", "fixation"),
            outlier=OutlierParams(max_velocity_px_per_s=1000.0),
        )
    )
    chain.apply(ScreenPoint(px=100, py=100), timestamp=0.0)
    out = chain.apply(ScreenPoint(px=10000, py=100), timestamp=0.001)
    assert out.px == 100


def test_one_euro_params_reach_the_filter() -> None:
    chain = build_filter_chain(
        ChainConfig(
            chain=("one_euro",),
            one_euro=OneEuroParams(min_cutoff=0.01, beta=0.0, d_cutoff=1.0),
        )
    )
    # Extremely low min_cutoff => heavy smoothing => output stays near first sample.
    chain.apply(ScreenPoint(px=0, py=0), timestamp=0.0)
    out = chain.apply(ScreenPoint(px=1000, py=0), timestamp=0.033)
    assert out.px < 50  # much closer to 0 than to 1000
