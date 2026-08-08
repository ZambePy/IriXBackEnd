"""ONNX Runtime backend for future full-graph exports.

Today's ``gaze_encoder.onnx`` is an **encoder** — it emits a 256-dim
embedding, not a 2-dim gaze prediction (MODEL_CARD.md §5). Using it as
the pipeline's gaze estimator would silently produce nonsense, so the
constructor inspects the output and refuses to load an encoder-only
graph.

The class stays here so the day a full ``gaze_cnn_best.onnx`` gets
exported (Sprint 6 R2 mitigation), the switch is one line in the YAML
config — matching the substitutability requirement from SPRINTS.md §7.5.

For direct access to the encoder's embedding output (parity comparison,
future distillation experiments) see :func:`embed_via_onnx` below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irisflow.core.exceptions import InferenceError
from irisflow.core.types import ModelInput, ModelMetadata, RawGaze
from irisflow.inference.base import add_batch_dim, clamp_gaze
from irisflow.logging import get_logger

__all__ = ["OnnxBackend", "embed_via_onnx", "onnx_output_kind"]


def _load_session(model_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise InferenceError(
            "ONNX backend requested but `onnxruntime` is not installed. "
            "Run: uv sync --extra inference-onnx"
        ) from exc
    if not model_path.exists():
        raise InferenceError(f"ONNX model not found at {model_path!r}.")
    try:
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise InferenceError(f"Failed to load ONNX model {model_path!r}: {exc}") from exc


def onnx_output_kind(session: Any) -> str:
    """Return ``"gaze_xy"``, ``"embedding"`` or ``"unknown"`` for ``session``.

    Splitting classification out of :class:`OnnxBackend`'s ``__init__``
    lets the parity tooling probe an encoder without instantiating a
    backend that would immediately reject it.
    """
    outputs = session.get_outputs()
    if len(outputs) != 1:
        return "unknown"
    shape = outputs[0].shape
    static_tail = [d for d in shape[1:] if isinstance(d, int)]
    if static_tail == [2]:
        return "gaze_xy"
    if len(static_tail) == 1 and static_tail[0] > 2:
        return "embedding"
    return "unknown"


class OnnxBackend:
    """:class:`GazeEstimator` over an ONNX Runtime session.

    Raises :class:`InferenceError` at construction if the loaded model is
    not a full estimator — see module docstring.
    """

    __slots__ = ("_input_names", "_log", "_metadata", "_session")

    def __init__(
        self,
        model_path: Path,
        *,
        channel_order: str = "RGB",
        normalization: str = "unit",
    ) -> None:
        session = _load_session(model_path)
        kind = onnx_output_kind(session)
        if kind != "gaze_xy":
            raise InferenceError(
                f"ONNX artifact {model_path.as_posix()!r} exposes a "
                f"{kind!r} output (shape={session.get_outputs()[0].shape!r}). "
                "OnnxBackend requires a full gaze estimator with output "
                "shape (batch, 2). The current release ships only the "
                "encoder — use the Keras backend for production."
            )
        input_names = tuple(inp.name for inp in session.get_inputs())
        self._session = session
        self._input_names = input_names
        self._metadata = ModelMetadata(
            backend="onnx",
            path=str(model_path),
            input_names=input_names,
            face_shape=(224, 224, 3),
            eye_shape=(112, 112, 3),
            rect_dim=12,
            channel_order=channel_order,
            normalization=normalization,
            output_kind="gaze_xy",
        )
        self._log = get_logger("irisflow.inference.onnx")

    def predict(self, model_input: ModelInput) -> RawGaze:
        import time

        batched = add_batch_dim(model_input)
        start = time.perf_counter()
        out = self._session.run(None, batched)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        arr = np.asarray(out[0])
        if arr.shape != (1, 2):
            raise InferenceError(f"ONNX model returned shape {arr.shape!r}, expected (1, 2).")
        return clamp_gaze(
            float(arr[0, 0]),
            float(arr[0, 1]),
            inference_ms=elapsed_ms,
            confidence=1.0,
            backend="onnx",
        )

    def warmup(self) -> None:
        dummy = {
            "face": np.zeros((1, 224, 224, 3), dtype=np.float32),
            "left_eye": np.zeros((1, 112, 112, 3), dtype=np.float32),
            "right_eye": np.zeros((1, 112, 112, 3), dtype=np.float32),
            "rect": np.zeros((1, 12), dtype=np.float32),
        }
        self._session.run(None, dummy)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata


def embed_via_onnx(model_path: Path, model_input: ModelInput) -> NDArray[np.float32]:
    """Run the encoder-only ONNX artifact and return the raw embedding.

    Only used by :mod:`irisflow.inference.parity`. Not exposed as a
    backend because gaze estimation requires the final Dense(2) that
    lives only in the Keras graph.
    """
    session = _load_session(model_path)
    kind = onnx_output_kind(session)
    if kind != "embedding":
        raise InferenceError(
            f"embed_via_onnx expected an encoder (embedding output), got "
            f"{kind!r} for {model_path.as_posix()!r}."
        )
    batched = add_batch_dim(model_input)
    out = session.run(None, batched)
    return np.asarray(out[0], dtype=np.float32)
