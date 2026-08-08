"""Pure geometry helpers used by detection, preprocessing and mapping.

Every function is a plain function of its arguments — no state, no I/O, no
dependency on anything but stdlib. Property-based tests in
``tests/unit/test_core_geometry.py`` pin down the invariants; this module
must remain trivial enough that a bug here is essentially impossible.

Coordinate conventions match :class:`~irisflow.core.types.BoundingBox`:
``x, y`` is the top-left corner; the frame lives in ``[0, W) x [0, H)``.
"""

from __future__ import annotations

from irisflow.core.types import BoundingBox

__all__ = [
    "clamp",
    "clip_to_frame",
    "denormalize",
    "expand",
    "normalize",
    "to_square",
]


def expand(bbox: BoundingBox, margin: float) -> BoundingBox:
    """Return ``bbox`` grown by ``margin`` (fraction of side length)."""
    return bbox.expand(margin)


def clip_to_frame(bbox: BoundingBox, frame_width: int, frame_height: int) -> BoundingBox:
    """Intersect ``bbox`` with the frame rectangle.

    Result may have ``w == 0`` or ``h == 0`` when the input lies fully
    outside the frame — callers must treat that as "no valid crop", not as a
    degenerate one-pixel box.
    """
    return bbox.clip(frame_width, frame_height)


def to_square(bbox: BoundingBox) -> BoundingBox:
    """Return the smallest square that fully contains ``bbox``, same center."""
    return bbox.to_square()


def normalize(
    bbox: BoundingBox, frame_width: int, frame_height: int
) -> tuple[float, float, float, float]:
    """Return ``(x, y, w, h)`` divided by frame dimensions."""
    return bbox.normalized(frame_width, frame_height)


def denormalize(nx: float, ny: float, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Turn a normalized ``(nx, ny) ∈ [0, 1]²`` back into pixel coordinates.

    Values outside the unit square are **not** clamped here — call
    :func:`clamp` explicitly if the caller wants that. Keeping this function
    pure lets tests distinguish "outside the screen" from "clamped to edge".
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"frame dims must be positive, got {frame_width}x{frame_height}")
    return round(nx * frame_width), round(ny * frame_height)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``. Fails loud on inverted bounds."""
    if low > high:
        raise ValueError(f"clamp bounds inverted: low={low} > high={high}")
    if value < low:
        return low
    if value > high:
        return high
    return value
