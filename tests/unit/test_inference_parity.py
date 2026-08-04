"""Sprint 6 — parity report math (uses fake estimators; no model needed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from irisflow.core.types import ModelInput, ModelMetadata, RawGaze
from irisflow.inference.parity import compare_gaze_backends


@dataclass
class _ScriptedEstimator:
    outputs: list[tuple[float, float]]
    _cursor: int = field(default=0)

    def predict(self, model_input: ModelInput) -> RawGaze:
        x, y = self.outputs[self._cursor % len(self.outputs)]
        self._cursor += 1
        return RawGaze(x=x, y=y, confidence=1.0, inference_ms=0.0)

    def warmup(self) -> None:
        return None

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            backend="scripted",
            path="<memory>",
            input_names=("face", "left_eye", "right_eye", "rect"),
            face_shape=(224, 224, 3),
            eye_shape=(112, 112, 3),
            rect_dim=12,
            channel_order="RGB",
            normalization="unit",
            output_kind="gaze_xy",
        )


def _mi() -> ModelInput:
    return ModelInput(
        face=np.zeros((224, 224, 3), dtype=np.float32),
        left_eye=np.zeros((112, 112, 3), dtype=np.float32),
        right_eye=np.zeros((112, 112, 3), dtype=np.float32),
        rect=np.zeros(12, dtype=np.float32),
    )


def test_identical_backends_produce_zero_diff() -> None:
    a = _ScriptedEstimator([(0.1, 0.9), (0.3, 0.7)])
    b = _ScriptedEstimator([(0.1, 0.9), (0.3, 0.7)])
    report = compare_gaze_backends(a, b, [_mi(), _mi()])
    assert report.max_abs_diff == pytest.approx(0.0)
    assert report.mean_abs_diff == pytest.approx(0.0)
    assert report.passed


def test_report_flags_failure_when_diff_exceeds_tolerance() -> None:
    a = _ScriptedEstimator([(0.5, 0.5)])
    b = _ScriptedEstimator([(0.6, 0.5)])  # 0.1 apart in x
    report = compare_gaze_backends(a, b, [_mi()], tolerance=1e-3)
    assert report.max_abs_diff == pytest.approx(0.1)
    assert not report.passed


def test_report_summarises_samples_and_mean() -> None:
    a = _ScriptedEstimator([(0.5, 0.5), (0.6, 0.6)])
    b = _ScriptedEstimator([(0.5, 0.6), (0.6, 0.5)])
    report = compare_gaze_backends(a, b, [_mi(), _mi()])
    assert report.samples == 2
    # Two pairs, each with one axis diff = 0.1 → mean over 4 numbers = 0.05.
    assert report.mean_abs_diff == pytest.approx(0.05)


def test_empty_inputs_raise() -> None:
    a = _ScriptedEstimator([(0.5, 0.5)])
    b = _ScriptedEstimator([(0.5, 0.5)])
    with pytest.raises(ValueError, match="at least one"):
        compare_gaze_backends(a, b, [])


def test_compare_keras_embedding_vs_onnx_smoke_missing_files(tmp_path: Path) -> None:
    """When files don't exist, the loader path raises a clear InferenceError.

    We don't want to import the real Keras/ONNX code paths in unit tests —
    covered separately by @pytest.mark.model integration tests once the
    artifacts are present.
    """
    from irisflow.core.exceptions import InferenceError
    from irisflow.inference.parity import compare_keras_embedding_vs_onnx

    with pytest.raises((InferenceError, ValueError)):
        compare_keras_embedding_vs_onnx(
            keras_model_path=tmp_path / "missing.keras",
            onnx_model_path=tmp_path / "missing.onnx",
            inputs=[_mi()],
        )
