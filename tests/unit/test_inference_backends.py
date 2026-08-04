"""Sprint 6 — real Keras / ONNX backends.

Split into two categories:

* Always-run: the "artifact missing" error paths (no model file needed).
* ``@pytest.mark.model``: fires up the real backends against the shipped
  artifacts. Skipped in CI unless the model files are present under
  ``models/`` (which they usually are on a developer's checkout).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irisflow.core.exceptions import InferenceError
from irisflow.core.types import ModelInput

_KERAS_MODEL = Path("models/gaze_cnn_best.keras")
_ONNX_MODEL = Path("models/gaze_encoder.onnx")

requires_keras = pytest.mark.skipif(
    not _KERAS_MODEL.exists(),
    reason=f"{_KERAS_MODEL} not present — skipping real-Keras tests",
)
requires_onnx = pytest.mark.skipif(
    not _ONNX_MODEL.exists(),
    reason=f"{_ONNX_MODEL} not present — skipping real-ONNX tests",
)


def _mi() -> ModelInput:
    return ModelInput(
        face=np.full((224, 224, 3), 0.5, dtype=np.float32),
        left_eye=np.full((112, 112, 3), 0.5, dtype=np.float32),
        right_eye=np.full((112, 112, 3), 0.5, dtype=np.float32),
        rect=np.array(
            [0.4, 0.3, 0.2, 0.4, 0.42, 0.42, 0.05, 0.03, 0.53, 0.42, 0.05, 0.03],
            dtype=np.float32,
        ),
    )


# ---------------------------------------------------------------------------
# Always-run: error paths
# ---------------------------------------------------------------------------
def test_keras_backend_reports_missing_model(tmp_path: Path) -> None:
    from irisflow.inference.keras_backend import KerasBackend

    with pytest.raises(InferenceError, match="Keras model not found"):
        KerasBackend(tmp_path / "does_not_exist.keras")


def test_onnx_backend_reports_missing_model(tmp_path: Path) -> None:
    from irisflow.inference.onnx_backend import OnnxBackend

    with pytest.raises(InferenceError, match="ONNX model not found"):
        OnnxBackend(tmp_path / "does_not_exist.onnx")


# ---------------------------------------------------------------------------
# Real Keras backend
# ---------------------------------------------------------------------------
@pytest.mark.model
@requires_keras
def test_real_keras_backend_predicts_in_unit_range() -> None:
    from irisflow.inference.keras_backend import KerasBackend

    backend = KerasBackend(_KERAS_MODEL, channel_order="RGB", normalization="unit")
    backend.warmup()
    raw = backend.predict(_mi())
    assert 0.0 <= raw.x <= 1.0
    assert 0.0 <= raw.y <= 1.0
    assert raw.inference_ms >= 0.0


@pytest.mark.model
@requires_keras
def test_real_keras_backend_metadata_matches_artifact() -> None:
    from irisflow.inference.keras_backend import KerasBackend

    backend = KerasBackend(_KERAS_MODEL, channel_order="RGB", normalization="unit")
    md = backend.metadata
    assert md.backend == "keras"
    assert md.face_shape == (224, 224, 3)
    assert md.eye_shape == (112, 112, 3)
    assert md.rect_dim == 12
    assert md.output_kind == "gaze_xy"
    assert set(md.input_names) == {"face", "left_eye", "right_eye", "rect"}


# ---------------------------------------------------------------------------
# Real ONNX backend — encoder-only artifact must be rejected clearly
# ---------------------------------------------------------------------------
@pytest.mark.model
@requires_onnx
def test_onnx_backend_rejects_encoder_only_artifact() -> None:
    from irisflow.inference.onnx_backend import OnnxBackend

    with pytest.raises(InferenceError, match="encoder"):
        OnnxBackend(_ONNX_MODEL, channel_order="RGB", normalization="unit")


@pytest.mark.model
@requires_onnx
def test_onnx_output_kind_classifies_encoder() -> None:
    from irisflow.inference.onnx_backend import _load_session, onnx_output_kind

    session = _load_session(_ONNX_MODEL)
    assert onnx_output_kind(session) == "embedding"


# ---------------------------------------------------------------------------
# Real parity: Keras embedding vs ONNX encoder should agree to 1e-3
# ---------------------------------------------------------------------------
@pytest.mark.model
@requires_keras
@requires_onnx
def test_real_parity_keras_embedding_matches_onnx_encoder() -> None:
    from irisflow.inference.parity import compare_keras_embedding_vs_onnx

    inputs = [_mi() for _ in range(3)]
    report = compare_keras_embedding_vs_onnx(
        _KERAS_MODEL, _ONNX_MODEL, inputs, tolerance=1e-3
    )
    assert report.passed, (
        f"Keras-embedding vs ONNX-encoder disagree: {report}"
    )
