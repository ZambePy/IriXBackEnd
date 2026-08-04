"""Sprint 4 — pure ROI helpers over synthetic landmark arrays."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from irisflow.detection.roi import bbox_from_indices


def _lms(points: list[tuple[float, float]]) -> np.ndarray:
    return np.array(points, dtype=np.float32)


def test_bbox_from_indices_returns_tight_box_around_points() -> None:
    lms = _lms([(10, 20), (30, 40), (25, 15), (5, 45)])
    box = bbox_from_indices(lms, (0, 1, 2, 3))
    # min x=5, min y=15, max x=30, max y=45 → w=25, h=30
    assert box.x == 5
    assert box.y == 15
    assert box.w == 25
    assert box.h == 30


def test_bbox_from_indices_uses_only_selected_indices() -> None:
    lms = _lms([(0, 0), (1000, 1000), (50, 50)])
    box = bbox_from_indices(lms, (0, 2))
    assert box.x == 0
    assert box.y == 0
    assert box.w == 50
    assert box.h == 50


def test_bbox_from_indices_expands_with_margin() -> None:
    lms = _lms([(20, 30), (40, 50)])
    box = bbox_from_indices(lms, (0, 1), margin=0.5)
    # Base box: x=20 y=30 w=20 h=20 → expand 0.5 → +10 each side
    assert box.w > 20
    assert box.h > 20


def test_bbox_from_indices_clips_to_frame() -> None:
    lms = _lms([(-10, -20), (500, 500)])
    box = bbox_from_indices(lms, (0, 1), frame_width=100, frame_height=100)
    assert 0 <= box.x <= 100
    assert 0 <= box.y <= 100
    assert box.x + box.w <= 100
    assert box.y + box.h <= 100


def test_bbox_from_indices_squares_when_requested() -> None:
    lms = _lms([(0, 0), (40, 10)])  # wide, short rectangle
    box = bbox_from_indices(lms, (0, 1), square=True)
    assert box.w == box.h


def test_bbox_from_indices_partial_out_of_frame_produces_valid_box() -> None:
    lms = _lms([(-30, 20), (50, 60)])
    box = bbox_from_indices(lms, (0, 1), frame_width=100, frame_height=100)
    assert box.x >= 0
    assert box.y >= 0
    # Some of the requested region is inside the frame — the box should
    # still cover the intersection.
    assert box.w > 0
    assert box.h > 0


def test_bbox_from_indices_rejects_empty_indices() -> None:
    lms = _lms([(0, 0), (10, 10)])
    with pytest.raises(ValueError, match="at least one"):
        bbox_from_indices(lms, ())


def test_bbox_from_indices_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r">=2"):
        bbox_from_indices(np.zeros((5,), dtype=np.float32), (0,))
    with pytest.raises(ValueError, match=r">=2"):
        bbox_from_indices(np.zeros((5, 1), dtype=np.float32), (0,))


@given(
    x0=st.integers(min_value=-500, max_value=500),
    y0=st.integers(min_value=-500, max_value=500),
    w=st.integers(min_value=1, max_value=500),
    h=st.integers(min_value=1, max_value=500),
    fw=st.integers(min_value=50, max_value=2000),
    fh=st.integers(min_value=50, max_value=2000),
)
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_bbox_from_indices_clipped_result_always_inside_frame(
    x0: int, y0: int, w: int, h: int, fw: int, fh: int
) -> None:
    lms = _lms([(x0, y0), (x0 + w, y0 + h)])
    box = bbox_from_indices(lms, (0, 1), frame_width=fw, frame_height=fh)
    assert 0 <= box.x <= fw
    assert 0 <= box.y <= fh
    assert 0 <= box.x + box.w <= fw
    assert 0 <= box.y + box.h <= fh
