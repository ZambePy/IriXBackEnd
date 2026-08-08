"""Sprint 5 — the 12-dim rect vector fed alongside the CNN's image tensors."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.core.types import BoundingBox
from irisflow.preprocessing.rect_vector import RECT_DIM, build_rect_vector


def _boxes_at(
    frame_w: int, frame_h: int, *, corner: str
) -> tuple[BoundingBox, BoundingBox, BoundingBox]:
    """Build face + eye boxes anchored to one of the four frame corners."""
    fw = frame_w // 4
    fh = frame_h // 4
    ew = fw // 3
    eh = fh // 3
    if corner == "top_left":
        fx, fy = 0, 0
    elif corner == "top_right":
        fx, fy = frame_w - fw, 0
    elif corner == "bottom_left":
        fx, fy = 0, frame_h - fh
    else:  # bottom_right
        fx, fy = frame_w - fw, frame_h - fh
    face = BoundingBox(x=fx, y=fy, w=fw, h=fh)
    left = BoundingBox(x=fx + fw // 4, y=fy + fh // 4, w=ew, h=eh)
    right = BoundingBox(x=fx + fw // 2, y=fy + fh // 4, w=ew, h=eh)
    return face, left, right


def test_rect_vector_has_correct_shape_and_dtype() -> None:
    face, left, right = _boxes_at(640, 480, corner="top_left")
    vec = build_rect_vector(face, left, right, frame_width=640, frame_height=480)
    assert vec.shape == (RECT_DIM,)
    assert vec.dtype == np.float32


def test_rect_vector_slots_match_documented_layout() -> None:
    face = BoundingBox(x=10, y=20, w=100, h=200)
    left = BoundingBox(x=30, y=60, w=10, h=8)
    right = BoundingBox(x=80, y=60, w=10, h=8)
    vec = build_rect_vector(face, left, right, frame_width=200, frame_height=400)
    # face indices 0..3
    assert vec[0] == pytest.approx(10 / 200)
    assert vec[1] == pytest.approx(20 / 400)
    assert vec[2] == pytest.approx(100 / 200)
    assert vec[3] == pytest.approx(200 / 400)
    # left eye 4..7
    assert vec[4] == pytest.approx(30 / 200)
    assert vec[5] == pytest.approx(60 / 400)
    assert vec[6] == pytest.approx(10 / 200)
    assert vec[7] == pytest.approx(8 / 400)
    # right eye 8..11
    assert vec[8] == pytest.approx(80 / 200)
    assert vec[9] == pytest.approx(60 / 400)


@pytest.mark.parametrize("corner", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_rect_values_in_unit_range_at_every_corner(corner: str) -> None:
    face, left, right = _boxes_at(1280, 720, corner=corner)
    vec = build_rect_vector(face, left, right, frame_width=1280, frame_height=720)
    assert vec.min() >= 0.0
    assert vec.max() <= 1.0


def test_rect_vector_writes_into_preallocated_out() -> None:
    face, left, right = _boxes_at(640, 480, corner="top_left")
    buf = np.zeros(RECT_DIM, dtype=np.float32)
    result = build_rect_vector(face, left, right, frame_width=640, frame_height=480, out=buf)
    assert result is buf
    assert buf.max() > 0  # actually populated


def test_reject_bad_out_shape() -> None:
    face, left, right = _boxes_at(640, 480, corner="top_left")
    with pytest.raises(ValueError, match="out must"):
        build_rect_vector(
            face,
            left,
            right,
            frame_width=640,
            frame_height=480,
            out=np.zeros(5, dtype=np.float32),
        )


def test_reject_non_positive_frame() -> None:
    face, left, right = _boxes_at(640, 480, corner="top_left")
    with pytest.raises(ValueError, match="positive"):
        build_rect_vector(face, left, right, frame_width=0, frame_height=480)
