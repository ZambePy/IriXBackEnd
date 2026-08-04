"""Sprint 6 — backend registry + warmup + custom-factory injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from irisflow.core.exceptions import ConfigError
from irisflow.core.types import ModelInput, ModelMetadata, RawGaze
from irisflow.inference.registry import build_gaze_estimator, register_backend
from irisflow.inference.warmup import warm_up_backend


@dataclass
class _FakeEstimator:
    path: Path
    channel_order: str = "RGB"
    normalization: str = "unit"
    warmup_calls: int = 0
    predict_calls: int = 0
    outputs: tuple[float, float] = (0.5, 0.5)

    def predict(self, model_input: ModelInput) -> RawGaze:
        self.predict_calls += 1
        return RawGaze(
            x=self.outputs[0],
            y=self.outputs[1],
            confidence=1.0,
            inference_ms=1.0,
        )

    def warmup(self) -> None:
        self.warmup_calls += 1

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            backend="fake",
            path=str(self.path),
            input_names=("face", "left_eye", "right_eye", "rect"),
            face_shape=(224, 224, 3),
            eye_shape=(112, 112, 3),
            rect_dim=12,
            channel_order=self.channel_order,
            normalization=self.normalization,
            output_kind="gaze_xy",
        )


@dataclass
class _Recorder:
    kwargs: dict = field(default_factory=dict)


def test_register_backend_wires_a_custom_factory() -> None:
    rec = _Recorder()

    def _factory(*, path: Path, channel_order: str, normalization: str) -> _FakeEstimator:
        rec.kwargs = {
            "path": path,
            "channel_order": channel_order,
            "normalization": normalization,
        }
        return _FakeEstimator(path=path, channel_order=channel_order, normalization=normalization)

    register_backend("test-fake", _factory)
    est = build_gaze_estimator(
        backend="test-fake",
        model_path=Path("nonexistent.keras"),
        channel_order="BGR",
        normalization="signed",
    )
    assert isinstance(est, _FakeEstimator)
    assert rec.kwargs["channel_order"] == "BGR"
    assert rec.kwargs["normalization"] == "signed"


def test_unknown_backend_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="Unknown inference backend"):
        build_gaze_estimator(
            backend="nope",
            model_path=Path("x"),
        )


def test_warm_up_backend_calls_warmup_n_times() -> None:
    register_backend("warmup-fake", lambda **kw: _FakeEstimator(path=kw["path"]))
    est = build_gaze_estimator(backend="warmup-fake", model_path=Path("x"))
    warm_up_backend(est, iterations=3)
    assert est.warmup_calls == 3  # type: ignore[attr-defined]


def test_warm_up_backend_rejects_negative_iterations() -> None:
    register_backend("warmup-neg", lambda **kw: _FakeEstimator(path=kw["path"]))
    est = build_gaze_estimator(backend="warmup-neg", model_path=Path("x"))
    with pytest.raises(ValueError, match="iterations"):
        warm_up_backend(est, iterations=-1)


def test_fake_estimator_conforms_to_gaze_estimator_protocol() -> None:
    from irisflow.core.interfaces import GazeEstimator

    est = _FakeEstimator(path=Path("x"))
    assert isinstance(est, GazeEstimator)

    mi = ModelInput(
        face=np.zeros((224, 224, 3), dtype=np.float32),
        left_eye=np.zeros((112, 112, 3), dtype=np.float32),
        right_eye=np.zeros((112, 112, 3), dtype=np.float32),
        rect=np.zeros(12, dtype=np.float32),
    )
    out = est.predict(mi)
    assert 0.0 <= out.x <= 1.0
    assert 0.0 <= out.y <= 1.0
