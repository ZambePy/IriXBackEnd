"""Landmarks → :class:`BoundingBox` — pure, no I/O, no MediaPipe.

Every function here is deterministic given its inputs, which makes them
trivially testable with synthetic landmark arrays and property-based tests
(Sprint 10 DoD).

The helpers deliberately do not know about "face" vs "eye" — the caller
passes the relevant :data:`~irisflow.detection.landmarks` index tuple.
Keeping this module ignorant of the domain vocabulary means it can be
reused unchanged for future ROIs (nose, mouth, brow) without edits.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from irisflow.core.types import BoundingBox

__all__ = ["bbox_from_indices"]


def bbox_from_indices(
    landmarks_px: NDArray[np.float32],
    indices: Sequence[int],
    *,
    margin: float = 0.0,
    frame_width: int | None = None,
    frame_height: int | None = None,
    square: bool = False,
) -> BoundingBox:
    """Compute the smallest integer :class:`BoundingBox` covering
    ``landmarks_px[indices]``, optionally expanded, squared and clipped.

    Args:
        landmarks_px: ``(N, >=2)`` array of pixel-space landmark coordinates.
        indices: Which rows to include.
        margin: Fractional expansion around the raw box (see
            :meth:`BoundingBox.expand`). ``0.2`` → 20% larger on each side.
        frame_width, frame_height: If both are set, clip the result to the
            frame rectangle so ``x``/``y`` are never negative and
            ``x + w <= frame_width`` (same for height).
        square: When true, expand the shorter side so the box is square
            (matches the CNN's crop shape — see SPRINTS.md §5).

    Order of operations is fixed: **expand → square → clip**. Clipping
    last means the returned box always fits inside the frame; squaring
    before clipping means the aspect ratio is fixed to 1:1 by the caller,
    even if the frame edge later trims one side.

    Raises:
        ValueError: ``indices`` is empty, or ``landmarks_px`` has the wrong
            shape.
    """
    if len(indices) == 0:
        raise ValueError("bbox_from_indices requires at least one index")
    if landmarks_px.ndim != 2 or landmarks_px.shape[1] < 2:
        raise ValueError(
            f"landmarks_px must be (N, >=2), got shape {landmarks_px.shape!r}"
        )

    selected = landmarks_px[list(indices), :2]
    xs = selected[:, 0]
    ys = selected[:, 1]

    x0 = int(np.floor(float(xs.min())))
    y0 = int(np.floor(float(ys.min())))
    x1 = int(np.ceil(float(xs.max())))
    y1 = int(np.ceil(float(ys.max())))

    box = BoundingBox(x=x0, y=y0, w=max(0, x1 - x0), h=max(0, y1 - y0))
    if margin > 0:
        box = box.expand(margin)
    if square:
        box = box.to_square()
    if frame_width is not None and frame_height is not None:
        box = box.clip(frame_width, frame_height)
    return box
