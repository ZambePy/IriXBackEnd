"""Shared helpers for every :class:`GazeEstimator` backend.

The two things every backend does the same way:

* Add a batch dimension of size 1 to each tensor in a :class:`ModelInput`,
  since the underlying frameworks always want ``(N, ...)`` inputs.
* Clamp the raw model output to ``[0, 1]²`` and emit a structured warning
  when it extrapolates — that condition is a signal to look for a
  channel-order or normalization bug (SPRINTS.md §7.3).

Kept as free functions (not an ABC) so tests can call them without
constructing a backend and so backends only have to implement
:meth:`predict` / :meth:`warmup` / :attr:`metadata`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irisflow.core.types import ModelInput, RawGaze
from irisflow.logging import get_logger

__all__ = ["EXPECTED_OUTPUT_RANGE", "add_batch_dim", "clamp_gaze"]


EXPECTED_OUTPUT_RANGE: tuple[float, float] = (0.0, 1.0)
"""Every gaze output is expected to be inside this range (see MODEL_CARD.md)."""


_log = get_logger("irisflow.inference.base")


def add_batch_dim(model_input: ModelInput) -> dict[str, NDArray[np.float32]]:
    """Return the four tensors keyed by name, each with a leading batch dim.

    The Keras and ONNX APIs both accept a ``dict[str, ndarray]``, and both
    use the same input names (face, left_eye, right_eye, rect — confirmed
    in MODEL_CARD.md). Sharing this helper keeps the two backends
    bit-for-bit consistent about how they present the batch.
    """
    return {
        "face": model_input.face[np.newaxis, ...],
        "left_eye": model_input.left_eye[np.newaxis, ...],
        "right_eye": model_input.right_eye[np.newaxis, ...],
        "rect": model_input.rect[np.newaxis, ...],
    }


def clamp_gaze(
    x: float,
    y: float,
    *,
    inference_ms: float,
    confidence: float = 1.0,
    backend: str = "unknown",
) -> RawGaze:
    """Clamp ``(x, y)`` into ``[0, 1]²`` and wrap into a :class:`RawGaze`.

    An extrapolating model is almost always a preprocessing bug (channel
    swap, wrong normalization). We clamp so the downstream pipeline sees
    valid coordinates *and* log a warning so the operator can go look at
    MODEL_CARD.md.
    """
    lo, hi = EXPECTED_OUTPUT_RANGE
    if not (lo <= x <= hi and lo <= y <= hi):
        _log.warning(
            "inference.output_extrapolated",
            backend=backend,
            raw_x=x,
            raw_y=y,
            expected=EXPECTED_OUTPUT_RANGE,
        )
    return RawGaze(
        x=float(min(max(x, lo), hi)),
        y=float(min(max(y, lo), hi)),
        confidence=float(confidence),
        inference_ms=float(inference_ms),
    )
