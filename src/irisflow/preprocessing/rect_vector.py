"""Construct the 12-dimensional ``rect`` auxiliary vector fed to the CNN.

**Canonical layout** (SPRINTS.md §7.4 — to be re-confirmed by Sprint 6):

======  ======================  ============
Index   Meaning                 Normalization
======  ======================  ============
0       ``face.x``              ``/ frame_width``
1       ``face.y``              ``/ frame_height``
2       ``face.w``              ``/ frame_width``
3       ``face.h``              ``/ frame_height``
4       ``left_eye.x``          ``/ frame_width``
5       ``left_eye.y``          ``/ frame_height``
6       ``left_eye.w``          ``/ frame_width``
7       ``left_eye.h``          ``/ frame_height``
8       ``right_eye.x``         ``/ frame_width``
9       ``right_eye.y``         ``/ frame_height``
10      ``right_eye.w``         ``/ frame_width``
11      ``right_eye.h``         ``/ frame_height``
======  ======================  ============

**Conventions**

* ``(x, y)`` refers to the **top-left corner** of the box, not its centre.
* All four coordinates are **relative to the full frame**, not to the face
  crop. This means the eye positions carry information about *where in
  the frame* the head is — useful signal for gaze estimation.
* "Left" and "right" refer to the **subject's** left and right (mirrored
  in a selfie preview), matching :mod:`irisflow.detection.landmarks`.

Output is a contiguous ``(12,)`` float32 array with every component in
``[0, 1]`` — the caller is expected to have already clipped the boxes to
the frame (see :meth:`~irisflow.core.types.BoundingBox.clip`), so no
extra clamping happens here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irisflow.core.types import BoundingBox

__all__ = ["RECT_DIM", "build_rect_vector"]


RECT_DIM = 12


def build_rect_vector(
    face_bbox: BoundingBox,
    left_eye_bbox: BoundingBox,
    right_eye_bbox: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Assemble the 12-dim rect vector for one face detection.

    Args:
        face_bbox, left_eye_bbox, right_eye_bbox: ROIs in pixel coordinates
            relative to the full frame.
        frame_width, frame_height: Dimensions of the full frame. Used as
            the normalization denominator — the same value produced by
            :attr:`Frame.width`/:attr:`Frame.height`.
        out: Optional preallocated ``(12,)`` float32 buffer. When passed,
            values are written into it and it is returned unchanged.

    Raises:
        ValueError: The frame dimensions are non-positive, or ``out`` has
            the wrong shape/dtype.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"frame dims must be positive, got {frame_width}x{frame_height}")
    if out is None:
        buf = np.empty(RECT_DIM, dtype=np.float32)
    else:
        if out.shape != (RECT_DIM,) or out.dtype != np.float32:
            raise ValueError(
                f"out must be ({RECT_DIM},) float32, got shape={out.shape!r} dtype={out.dtype}"
            )
        buf = out

    fw = float(frame_width)
    fh = float(frame_height)
    for slot, box in enumerate((face_bbox, left_eye_bbox, right_eye_bbox)):
        base = slot * 4
        buf[base + 0] = box.x / fw
        buf[base + 1] = box.y / fh
        buf[base + 2] = box.w / fw
        buf[base + 3] = box.h / fh
    return buf
