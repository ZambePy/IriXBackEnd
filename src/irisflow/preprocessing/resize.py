"""Resize an image to a target ``(H, W)``.

Interpolation choice matters. OpenCV documents these trade-offs:

* ``INTER_AREA`` — best when **downscaling**. Averages source pixels so
  aliasing/moiré doesn't survive the shrink. Slightly slower than linear.
* ``INTER_LINEAR`` — best when **upscaling** or resizing by a small factor.
  Fast, produces smooth output.
* ``INTER_CUBIC`` / ``INTER_LANCZOS4`` — sharper but slower; only useful
  for photo-quality resizes, never in a real-time loop.

If the training script for the CNN used a specific interpolation, we must
match it exactly (Sprint 6 gate). Until MODEL_CARD confirms, this module
picks ``INTER_AREA`` for down and ``INTER_LINEAR`` for up-scaling — the
most common convention.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["Interpolation", "resize_to"]


Interpolation = Literal["auto", "linear", "area", "cubic", "nearest"]


_INTER_MAP: dict[str, int] = {
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


def resize_to(
    image: NDArray[np.uint8],
    target_hw: tuple[int, int],
    *,
    interpolation: Interpolation = "auto",
    dst: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Resize ``image`` to ``(target_h, target_w)`` using the chosen filter.

    Args:
        image: ``(H, W, 3)`` uint8 source.
        target_hw: ``(target_h, target_w)`` — the desired output shape,
            same convention as :attr:`ndarray.shape` (NOT ``(w, h)``).
        interpolation: See module docstring. ``"auto"`` picks
            ``INTER_AREA`` when the output pixel count is smaller than the
            input, ``INTER_LINEAR`` otherwise.
        dst: Optional preallocated ``(target_h, target_w, 3)`` uint8
            buffer. When passed, ``cv2.resize`` writes into it and no
            allocation occurs in the hot loop.

    Raises:
        ValueError: The target size is non-positive, or ``dst`` has the
            wrong shape/dtype.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"resize_to expects (H, W, 3), got {image.shape!r}")
    target_h, target_w = target_hw
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_hw must be positive, got {target_hw!r}")

    if interpolation == "auto":
        source_pixels = image.shape[0] * image.shape[1]
        target_pixels = target_h * target_w
        inter = cv2.INTER_AREA if target_pixels < source_pixels else cv2.INTER_LINEAR
    else:
        inter = _INTER_MAP[interpolation]

    if dst is not None:
        if dst.shape != (target_h, target_w, 3) or dst.dtype != np.uint8:
            raise ValueError(
                f"dst must be ({target_h},{target_w},3) uint8, got "
                f"shape={dst.shape!r} dtype={dst.dtype}"
            )
        cv2.resize(image, (target_w, target_h), dst=dst, interpolation=inter)
        return dst
    resized = cv2.resize(image, (target_w, target_h), interpolation=inter)
    return np.ascontiguousarray(resized, dtype=np.uint8)
