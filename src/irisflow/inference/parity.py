"""Compare two inference backends on the same inputs.

Sprint 6 requires Keras/ONNX parity or a documented explanation. Since
the shipped ``gaze_encoder.onnx`` is an encoder (256-dim embedding, not
gaze), the strict apples-to-apples comparison would compare Keras'
``embedding`` layer output with the ONNX output — same tensor, two
runtimes.

This module exposes helpers for both flavours:

* :func:`compare_gaze_backends` — two full gaze backends → per-axis max
  absolute error over a batch of ModelInputs.
* :func:`compare_keras_embedding_vs_onnx` — Keras' embedding layer vs
  the encoder-only ONNX; the practical parity check for today's ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from irisflow.core.interfaces import GazeEstimator

from irisflow.core.exceptions import InferenceError
from irisflow.core.types import ModelInput
from irisflow.inference.base import add_batch_dim
from irisflow.inference.onnx_backend import embed_via_onnx

__all__ = [
    "ParityReport",
    "compare_gaze_backends",
    "compare_keras_embedding_vs_onnx",
]


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Summary of a parity comparison between two backends."""

    samples: int
    max_abs_diff: float
    mean_abs_diff: float
    tolerance: float
    passed: bool


def compare_gaze_backends(
    lhs: GazeEstimator,
    rhs: GazeEstimator,
    inputs: list[ModelInput],
    *,
    tolerance: float = 1e-3,
) -> ParityReport:
    """Run both estimators on ``inputs`` and report the disagreement.

    Args:
        lhs, rhs: Two :class:`GazeEstimator` implementations expected to
            produce the same output.
        inputs: The set of ``ModelInput``s to score. Ordering matters —
            per-sample errors are matched positionally.
        tolerance: Max allowed absolute per-axis error before ``passed``
            flips to ``False``. ``1e-3`` = 0.1% of the normalized range.
    """
    if not inputs:
        raise ValueError("compare_gaze_backends requires at least one input")

    diffs: list[float] = []
    for mi in inputs:
        a = lhs.predict(mi)
        b = rhs.predict(mi)
        diffs.append(abs(a.x - b.x))
        diffs.append(abs(a.y - b.y))
    arr = np.asarray(diffs, dtype=np.float64)
    max_diff = float(arr.max())
    mean_diff = float(arr.mean())
    return ParityReport(
        samples=len(inputs),
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        tolerance=tolerance,
        passed=max_diff < tolerance,
    )


def compare_keras_embedding_vs_onnx(
    keras_model_path: Path,
    onnx_model_path: Path,
    inputs: list[ModelInput],
    *,
    tolerance: float = 1e-3,
) -> ParityReport:
    """Compare Keras' ``embedding`` layer output with the ONNX encoder output.

    The practical parity check while the ONNX file is encoder-only.
    """
    if not inputs:
        raise ValueError("compare_keras_embedding_vs_onnx requires >= 1 input")
    keras_embeddings = _keras_embeddings(keras_model_path, inputs)
    onnx_embeddings = _onnx_embeddings(onnx_model_path, inputs)
    if keras_embeddings.shape != onnx_embeddings.shape:
        raise InferenceError(
            "Embedding shape mismatch: Keras "
            f"{keras_embeddings.shape!r} vs ONNX {onnx_embeddings.shape!r}"
        )
    diffs = np.abs(keras_embeddings - onnx_embeddings)
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())
    return ParityReport(
        samples=len(inputs),
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        tolerance=tolerance,
        passed=max_diff < tolerance,
    )


def _keras_embeddings(
    keras_model_path: Path, inputs: list[ModelInput]
) -> NDArray[np.float32]:
    import keras

    model = keras.models.load_model(keras_model_path, compile=False)
    try:
        emb_layer = model.get_layer("embedding")
    except ValueError as exc:
        raise InferenceError(
            "Keras model has no 'embedding' layer — cannot compare "
            "against the ONNX encoder without a shared cut point."
        ) from exc
    trunk = keras.Model(inputs=model.inputs, outputs=emb_layer.output)
    out = np.stack(
        [np.asarray(trunk(add_batch_dim(mi), training=False))[0] for mi in inputs]
    )
    return out.astype(np.float32)


def _onnx_embeddings(
    onnx_model_path: Path, inputs: list[ModelInput]
) -> NDArray[np.float32]:
    return np.stack(
        [embed_via_onnx(onnx_model_path, mi)[0] for mi in inputs]
    ).astype(np.float32)
