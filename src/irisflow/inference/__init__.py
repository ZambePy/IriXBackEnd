"""Inference layer — the CNN motor and its runtime backends.

Public surface:

* :func:`build_gaze_estimator` — the one call the pipeline needs.
* :class:`KerasBackend`, :class:`OnnxBackend` — concrete implementations
  of :class:`~irisflow.core.interfaces.GazeEstimator`.
* :func:`warm_up_backend` — call at startup to eat the JIT stall.
* :mod:`parity` — Keras vs ONNX cross-check tooling.
"""

from irisflow.inference.parity import (
    ParityReport,
    compare_gaze_backends,
    compare_keras_embedding_vs_onnx,
)
from irisflow.inference.registry import (
    BackendName,
    build_gaze_estimator,
    register_backend,
)
from irisflow.inference.warmup import warm_up_backend

__all__ = [
    "BackendName",
    "ParityReport",
    "build_gaze_estimator",
    "compare_gaze_backends",
    "compare_keras_embedding_vs_onnx",
    "register_backend",
    "warm_up_backend",
]
