"""Crop a rectangular ROI from a full frame, padding with edge replication
when the box extends outside the image.

Why replication (not zero-padding): a black border in the crop is a strong
visual feature that the CNN has never seen — face-out-of-frame training
data doesn't include a rectangle of pure black. Replicating the nearest
edge pixels keeps the crop statistics closer to the training distribution.

The function returns a **contiguous** ``uint8`` ndarray of shape
``(bbox.h, bbox.w, 3)``. Callers can safely pass the result to
:mod:`irisflow.preprocessing.resize` and :mod:`irisflow.preprocessing.normalize`
without an intermediate copy.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from irisflow.core.types import BoundingBox

__all__ = ["crop_with_replicate_pad"]


def crop_with_replicate_pad(
    image: NDArray[np.uint8],
    bbox: BoundingBox,
) -> NDArray[np.uint8]:
    """Return the ``bbox`` region of ``image`` padded with replicated edges.

    Args:
        image: ``(H, W, 3)`` uint8 BGR/RGB frame — the function doesn't
            interpret the channel order, it only slices pixels.
        bbox: The region to extract. May be entirely inside, partly outside,
            or fully outside the image; only the fully-outside case raises.

    Raises:
        ValueError: The bbox has non-positive width/height, or its
            intersection with the frame is empty (fully outside).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"crop_with_replicate_pad expects (H, W, 3), got {image.shape!r}"
        )
    if bbox.w <= 0 or bbox.h <= 0:
        raise ValueError(f"bbox has non-positive size: w={bbox.w} h={bbox.h}")

    frame_h, frame_w = image.shape[:2]
    valid_x1 = max(0, bbox.x)
    valid_y1 = max(0, bbox.y)
    valid_x2 = min(frame_w, bbox.x2)
    valid_y2 = min(frame_h, bbox.y2)

    if valid_x2 <= valid_x1 or valid_y2 <= valid_y1:
        raise ValueError(
            f"bbox {bbox!r} has no intersection with frame {frame_w}x{frame_h}"
        )

    pad_left = valid_x1 - bbox.x
    pad_top = valid_y1 - bbox.y
    pad_right = bbox.x2 - valid_x2
    pad_bottom = bbox.y2 - valid_y2

    valid = image[valid_y1:valid_y2, valid_x1:valid_x2]
    if pad_left == 0 and pad_top == 0 and pad_right == 0 and pad_bottom == 0:
        # Contiguous copy so the caller can safely pass to `cv2.resize(dst=...)`.
        return np.ascontiguousarray(valid)
    padded = cv2.copyMakeBorder(
        valid,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    return np.ascontiguousarray(padded, dtype=np.uint8)
