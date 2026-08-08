"""Named MediaPipe FaceMesh landmark indices — no magic numbers elsewhere.

MediaPipe returns 478 face landmarks when the model includes iris refinement
(indices 468-477); older/simpler models return 468 (no iris).

**Handedness convention.** Throughout the codebase, "left" and "right" refer
to the **subject's** left and right — the natural anatomy — not the pixel
position in the (possibly mirrored) camera image. The subject's left eye
appears on the right side of a selfie-style preview. This is the same
convention MediaPipe itself uses.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FACE_LANDMARK_COUNT_LEGACY",
    "FACE_LANDMARK_COUNT_WITH_IRIS",
    "FACE_OVAL_INDICES",
    "LEFT_EYE_INDICES",
    "LEFT_IRIS_INDICES",
    "RIGHT_EYE_INDICES",
    "RIGHT_IRIS_INDICES",
    "landmarks_to_pixel_array",
]


FACE_LANDMARK_COUNT_LEGACY: Final[int] = 468
FACE_LANDMARK_COUNT_WITH_IRIS: Final[int] = 478


# Iris landmarks — only present when refine_landmarks / an iris-capable
# model is used. Five points each, arranged as centre + four rim.
LEFT_IRIS_INDICES: Final[tuple[int, ...]] = (468, 469, 470, 471, 472)
RIGHT_IRIS_INDICES: Final[tuple[int, ...]] = (473, 474, 475, 476, 477)


# Curated eye-contour subsets — dense enough to bound each eye tightly but
# small enough to keep bbox computation cheap. Indices sourced from
# MediaPipe's ``FACEMESH_LEFT_EYE`` / ``FACEMESH_RIGHT_EYE`` connection sets.
LEFT_EYE_INDICES: Final[tuple[int, ...]] = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
)
RIGHT_EYE_INDICES: Final[tuple[int, ...]] = (
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
)


# Face contour (FACEMESH_FACE_OVAL). Bounds the whole face reliably even
# when the head is turned — better than using the entire landmark set,
# which includes noisy interior points.
FACE_OVAL_INDICES: Final[tuple[int, ...]] = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)


def landmarks_to_pixel_array(
    normalized: NDArray[np.float32],
    frame_width: int,
    frame_height: int,
) -> NDArray[np.float32]:
    """Scale a ``(N, 3)`` normalized-landmark array into pixel space.

    Input is what MediaPipe returns: ``x``/``y`` in ``[0, 1]``, ``z`` in
    world units (mostly unused here). Output uses the same shape and
    dtype, with ``x`` multiplied by ``frame_width`` and ``y`` by
    ``frame_height`` — ``z`` is left untouched.
    """
    if normalized.ndim != 2 or normalized.shape[1] < 2:
        raise ValueError(f"landmarks_to_pixel_array expects (N, >=2), got {normalized.shape!r}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"frame dims must be positive, got {frame_width}x{frame_height}")
    out = normalized.astype(np.float32, copy=True)
    out[:, 0] *= float(frame_width)
    out[:, 1] *= float(frame_height)
    return out
