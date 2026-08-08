"""Keras-backed :class:`GazeEstimator` — the production motor.

`gaze_cnn_best.keras` is the only artifact that emits a full
`(gaze_x, gaze_y)` prediction (see :file:`models/MODEL_CARD.md`), so this
is the default backend in every deployment. The ONNX file is an encoder
and cannot serve here.

Keras + TensorFlow are heavy dependencies. Imports are deferred so
production code that doesn't touch inference (tests, capture, doctor)
doesn't pay for them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from irisflow.core.exceptions import InferenceError
from irisflow.core.types import ModelInput, ModelMetadata, RawGaze
from irisflow.inference.base import add_batch_dim, clamp_gaze
from irisflow.logging import get_logger

# Silence the TF C++ banner unconditionally when this module is imported.
# The user can override by setting TF_CPP_MIN_LOG_LEVEL before importing.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


__all__ = ["KerasBackend"]


class KerasBackend:
    """:class:`GazeEstimator` implementation over a Keras model file.

    Args:
        model_path: Path to a ``.keras`` archive (Keras 3 format).
        channel_order: Kept for :class:`ModelMetadata` — the actual
            channel-order handling happens in preprocessing.
        normalization: Same — recorded here for traceability.
    """

    __slots__ = ("_log", "_metadata", "_model", "_predict_fn")

    def __init__(
        self,
        model_path: Path,
        *,
        channel_order: str = "RGB",
        normalization: str = "unit",
    ) -> None:
        if not model_path.exists():
            raise InferenceError(
                f"Keras model not found at {model_path!r}. "
                "Download / copy the artifact before starting inference."
            )
        # Deferred heavy import.
        try:
            import keras
        except ImportError as exc:  # pragma: no cover
            raise InferenceError(
                "Keras backend requested but `keras` is not installed. "
                "Run: uv sync --extra inference-keras"
            ) from exc

        try:
            model = keras.models.load_model(model_path, compile=False)
        except Exception as exc:
            raise InferenceError(f"Failed to load Keras model {model_path!r}: {exc}") from exc

        self._model = model
        input_names = tuple(inp.name for inp in model.inputs)
        output_shape = tuple(model.outputs[0].shape)
        if len(output_shape) != 2 or output_shape[-1] != 2:
            raise InferenceError(
                f"Loaded Keras model has output shape {output_shape!r}, "
                "expected (batch, 2). This artifact is not a gaze estimator."
            )
        # `keras.Model.__call__` is faster and lower-overhead than
        # `.predict()` for single-sample inference; use it directly.
        self._predict_fn = model
        self._metadata = ModelMetadata(
            backend="keras",
            path=str(model_path),
            input_names=input_names,
            face_shape=(224, 224, 3),
            eye_shape=(112, 112, 3),
            rect_dim=12,
            channel_order=channel_order,
            normalization=normalization,
            output_kind="gaze_xy",
        )
        self._log = get_logger("irisflow.inference.keras")

    # ------------------------------------------------------------------ API
    def predict(self, model_input: ModelInput) -> RawGaze:
        import time

        batched = add_batch_dim(model_input)
        start = time.perf_counter()
        out = self._predict_fn(batched, training=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        arr = _to_numpy(out)
        if arr.shape != (1, 2):
            raise InferenceError(f"Keras model returned shape {arr.shape!r}, expected (1, 2).")
        return clamp_gaze(
            float(arr[0, 0]),
            float(arr[0, 1]),
            inference_ms=elapsed_ms,
            confidence=1.0,
            backend="keras",
        )

    def warmup(self) -> None:
        # The first `__call__` on a fresh Keras model compiles a graph;
        # skipping warmup would make the first real frame ~10x slower.
        import numpy as np

        dummy = {
            "face": np.zeros((1, 224, 224, 3), dtype=np.float32),
            "left_eye": np.zeros((1, 112, 112, 3), dtype=np.float32),
            "right_eye": np.zeros((1, 112, 112, 3), dtype=np.float32),
            "rect": np.zeros((1, 12), dtype=np.float32),
        }
        self._predict_fn(dummy, training=False)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata


def _to_numpy(out: Any) -> Any:
    """Normalize whatever Keras returns (numpy array / TF tensor / list)
    to a numpy array with shape ``(1, 2)``."""
    if hasattr(out, "numpy"):
        return out.numpy()
    if isinstance(out, list):
        return _to_numpy(out[0])
    return out
