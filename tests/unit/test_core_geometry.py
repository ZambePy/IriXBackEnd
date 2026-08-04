"""Sprint 2 — geometry helpers.

100% coverage target (SPRINTS.md §S2). Property-based tests via hypothesis
pin down the invariants; the unit cases nail the tricky edges: boxes fully
outside the frame, ``w`` or ``h`` == 0, negative coordinates.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from irisflow.core.geometry import (
    clamp,
    clip_to_frame,
    denormalize,
    expand,
    normalize,
    to_square,
)
from irisflow.core.types import BoundingBox


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------
def test_expand_grows_symmetrically_around_center() -> None:
    box = BoundingBox(x=100, y=100, w=100, h=100)
    grown = expand(box, 0.2)  # +20 px total
    assert (grown.w, grown.h) == (120, 120)
    assert (grown.x, grown.y) == (90, 90)  # center at (150, 150) preserved


def test_expand_zero_margin_is_a_noop() -> None:
    box = BoundingBox(x=10, y=20, w=30, h=40)
    assert expand(box, 0.0) == box


def test_expand_rejects_negative_margin() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        expand(BoundingBox(0, 0, 10, 10), -0.1)


def test_expand_on_zero_size_box_stays_zero() -> None:
    box = BoundingBox(x=5, y=5, w=0, h=0)
    assert expand(box, 0.5) == box


# ---------------------------------------------------------------------------
# clip_to_frame
# ---------------------------------------------------------------------------
def test_clip_leaves_interior_box_untouched() -> None:
    box = BoundingBox(x=10, y=10, w=50, h=50)
    assert clip_to_frame(box, 640, 480) == box


def test_clip_shrinks_box_crossing_the_right_edge() -> None:
    box = BoundingBox(x=600, y=10, w=100, h=50)
    clipped = clip_to_frame(box, 640, 480)
    assert (clipped.x, clipped.y, clipped.w, clipped.h) == (600, 10, 40, 50)


def test_clip_shrinks_box_crossing_the_top_left_corner() -> None:
    box = BoundingBox(x=-20, y=-30, w=100, h=100)
    clipped = clip_to_frame(box, 640, 480)
    assert (clipped.x, clipped.y, clipped.w, clipped.h) == (0, 0, 80, 70)


def test_clip_returns_empty_box_when_fully_outside() -> None:
    box = BoundingBox(x=700, y=700, w=50, h=50)
    clipped = clip_to_frame(box, 640, 480)
    assert clipped.w == 0
    assert clipped.h == 0


def test_clip_of_negative_size_is_empty() -> None:
    box = BoundingBox(x=10, y=10, w=-5, h=-5)
    clipped = clip_to_frame(box, 640, 480)
    assert clipped.area == 0


# ---------------------------------------------------------------------------
# to_square
# ---------------------------------------------------------------------------
def test_to_square_expands_shorter_side() -> None:
    box = BoundingBox(x=100, y=100, w=80, h=40)
    square = to_square(box)
    assert square.w == square.h == 80
    # center preserved
    assert square.x + square.w / 2 == pytest.approx(box.x + box.w / 2)
    assert square.y + square.h / 2 == pytest.approx(box.y + box.h / 2)


def test_to_square_is_idempotent_for_squares() -> None:
    box = BoundingBox(x=0, y=0, w=50, h=50)
    assert to_square(box) == box


def test_to_square_of_zero_box_is_zero_box() -> None:
    box = BoundingBox(x=5, y=5, w=0, h=0)
    assert to_square(box) == box


# ---------------------------------------------------------------------------
# normalize / denormalize
# ---------------------------------------------------------------------------
def test_normalize_returns_fractions_of_frame_dims() -> None:
    box = BoundingBox(x=320, y=240, w=160, h=120)
    assert normalize(box, 640, 480) == (0.5, 0.5, 0.25, 0.25)


def test_normalize_rejects_zero_frame_dims() -> None:
    box = BoundingBox(0, 0, 10, 10)
    with pytest.raises(ValueError, match="frame dims"):
        normalize(box, 0, 480)
    with pytest.raises(ValueError, match="frame dims"):
        normalize(box, 640, 0)


def test_denormalize_rounds_to_nearest_pixel() -> None:
    assert denormalize(0.5, 0.25, 640, 480) == (320, 120)
    assert denormalize(0.999, 0.001, 100, 100) == (100, 0)


def test_denormalize_does_not_clamp_out_of_range_values() -> None:
    # Callers can clamp() themselves; keeping this function pure lets tests
    # distinguish "outside the screen" from "clamped to the edge".
    assert denormalize(1.5, -0.2, 1000, 1000) == (1500, -200)


def test_denormalize_rejects_zero_frame_dims() -> None:
    with pytest.raises(ValueError, match="frame dims"):
        denormalize(0.5, 0.5, 0, 100)
    with pytest.raises(ValueError, match="frame dims"):
        denormalize(0.5, 0.5, 100, 0)


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [
        (5.0, 0.0, 10.0, 5.0),
        (-1.0, 0.0, 10.0, 0.0),
        (11.0, 0.0, 10.0, 10.0),
        (10.0, 10.0, 10.0, 10.0),
    ],
)
def test_clamp_bounds_the_value(value: float, low: float, high: float, expected: float) -> None:
    assert clamp(value, low, high) == expected


def test_clamp_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="inverted"):
        clamp(5.0, 10.0, 0.0)


# ---------------------------------------------------------------------------
# Property-based: clip always yields a box inside the frame
# ---------------------------------------------------------------------------
@given(
    x=st.integers(min_value=-1000, max_value=1000),
    y=st.integers(min_value=-1000, max_value=1000),
    w=st.integers(min_value=-100, max_value=1000),
    h=st.integers(min_value=-100, max_value=1000),
    fw=st.integers(min_value=1, max_value=4000),
    fh=st.integers(min_value=1, max_value=4000),
)
def test_property_clip_result_is_inside_frame(
    x: int, y: int, w: int, h: int, fw: int, fh: int
) -> None:
    clipped = clip_to_frame(BoundingBox(x=x, y=y, w=w, h=h), fw, fh)
    assert 0 <= clipped.x <= fw
    assert 0 <= clipped.y <= fh
    assert clipped.x + clipped.w <= fw
    assert clipped.y + clipped.h <= fh
    assert clipped.w >= 0
    assert clipped.h >= 0


@given(
    x=st.integers(min_value=0, max_value=500),
    y=st.integers(min_value=0, max_value=500),
    w=st.integers(min_value=1, max_value=500),
    h=st.integers(min_value=1, max_value=500),
    fw=st.integers(min_value=100, max_value=2000),
    fh=st.integers(min_value=100, max_value=2000),
)
def test_property_normalize_and_denormalize_roundtrip_within_one_pixel(
    x: int, y: int, w: int, h: int, fw: int, fh: int
) -> None:
    box = BoundingBox(x=x, y=y, w=w, h=h)
    nx, ny, _, _ = normalize(box, fw, fh)
    px, py = denormalize(nx, ny, fw, fh)
    assert abs(px - x) <= 1
    assert abs(py - y) <= 1


@given(
    value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False),
    low=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    span=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
)
def test_property_clamp_result_is_within_bounds(
    value: float, low: float, span: float
) -> None:
    high = low + span
    result = clamp(value, low, high)
    assert low <= result <= high
