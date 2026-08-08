"""Sprint 8 -- calibration models: affine, polynomial, ridge.

The core criterion (§8 DoD): "the transformation recovers a known
injected bias within tolerance". The bias-injection helper below is
the shared setup: we generate targets, produce raw samples by applying
a known transform, feed them to the model, and verify the model
transforms new raw samples back to the target space.
"""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.calibration.models import (
    AffineCalibration,
    PolynomialCalibration,
    RidgeCalibration,
    build_calibration_model,
)
from irisflow.calibration.points import generate_targets
from irisflow.core.exceptions import CalibrationError
from irisflow.core.types import RawGaze


def _samples_with_affine_bias(
    scale_x: float = 1.1,
    scale_y: float = 0.9,
    offset_x: float = 0.05,
    offset_y: float = -0.03,
    per_target: int = 10,
) -> tuple[list[RawGaze], list[tuple[float, float]]]:
    """Produce raw samples that = scale*(target) + offset + tiny noise."""
    rng = np.random.default_rng(42)
    targets = generate_targets(9, margin=0.1)
    raw: list[RawGaze] = []
    tgt: list[tuple[float, float]] = []
    for t in targets:
        for _ in range(per_target):
            noise_x = float(rng.normal(0.0, 0.002))
            noise_y = float(rng.normal(0.0, 0.002))
            raw.append(
                RawGaze(
                    x=t.nx * scale_x + offset_x + noise_x,
                    y=t.ny * scale_y + offset_y + noise_y,
                    confidence=1.0,
                    inference_ms=1.0,
                )
            )
            tgt.append((t.nx, t.ny))
    return raw, tgt


# ---------------------------------------------------------------------------
# Affine
# ---------------------------------------------------------------------------
def test_affine_recovers_known_bias() -> None:
    raw, tgt = _samples_with_affine_bias()
    model = AffineCalibration()
    model.fit(raw, tgt)
    assert model.is_fitted
    # For a hold-out sample, the calibrated output must be within 0.02 of
    # the true target -- the noise floor is ~0.002, model is exact.
    test_raw = RawGaze(x=0.6 * 1.1 + 0.05, y=0.4 * 0.9 - 0.03, confidence=1.0, inference_ms=1.0)
    out = model.transform(test_raw)
    assert out.x == pytest.approx(0.6, abs=0.02)
    assert out.y == pytest.approx(0.4, abs=0.02)


def test_affine_transform_before_fit_raises() -> None:
    model = AffineCalibration()
    with pytest.raises(CalibrationError, match="before fit"):
        model.transform(RawGaze(x=0.5, y=0.5, confidence=1.0, inference_ms=1.0))


def test_affine_fit_rejects_mismatched_lengths() -> None:
    model = AffineCalibration()
    raw = [RawGaze(x=0.1, y=0.1, confidence=1.0, inference_ms=1.0)]
    with pytest.raises(CalibrationError, match="raw samples vs"):
        model.fit(raw, [(0.1, 0.1), (0.2, 0.2)])


def test_affine_fit_rejects_empty() -> None:
    with pytest.raises(CalibrationError, match="at least one"):
        AffineCalibration().fit([], [])


def test_affine_fit_rejects_underdetermined() -> None:
    raw = [RawGaze(x=0.1, y=0.1, confidence=1.0, inference_ms=1.0)]
    with pytest.raises(CalibrationError, match="need at least"):
        AffineCalibration().fit(raw, [(0.5, 0.5)])


def test_affine_rejects_nonfinite_input() -> None:
    raw = [RawGaze(x=float("nan"), y=0.1, confidence=1.0, inference_ms=1.0)] * 5
    tgt = [(0.5, 0.5)] * 5
    with pytest.raises(CalibrationError, match="NaN or inf"):
        AffineCalibration().fit(raw, tgt)


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------
def test_polynomial_recovers_second_order_bias() -> None:
    """Polynomial can fit a distortion no affine can: t = x + 0.1*x^2."""
    targets = generate_targets(9, margin=0.1)
    raw: list[RawGaze] = []
    tgt: list[tuple[float, float]] = []
    rng = np.random.default_rng(7)
    for t in targets:
        for _ in range(20):
            noise = float(rng.normal(0.0, 0.001))
            raw.append(
                RawGaze(
                    x=t.nx + 0.1 * t.nx * t.nx + noise,
                    y=t.ny - 0.05 * t.ny * t.ny + noise,
                    confidence=1.0,
                    inference_ms=1.0,
                )
            )
            tgt.append((t.nx, t.ny))
    model = PolynomialCalibration()
    model.fit(raw, tgt)
    for t in targets:
        r = RawGaze(
            x=t.nx + 0.1 * t.nx * t.nx,
            y=t.ny - 0.05 * t.ny * t.ny,
            confidence=1.0,
            inference_ms=1.0,
        )
        out = model.transform(r)
        assert out.x == pytest.approx(t.nx, abs=0.01)
        assert out.y == pytest.approx(t.ny, abs=0.01)


def test_polynomial_only_supports_degree_2() -> None:
    with pytest.raises(CalibrationError, match="degree=2"):
        PolynomialCalibration(degree=3)


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------
def test_ridge_reduces_overfit_with_few_samples() -> None:
    """With only 2 samples per target the ridge fit remains stable."""
    targets = generate_targets(9, margin=0.1)
    raw: list[RawGaze] = []
    tgt: list[tuple[float, float]] = []
    for t in targets:
        for _ in range(2):
            raw.append(
                RawGaze(
                    x=t.nx * 1.05 + 0.02,
                    y=t.ny * 0.95 - 0.02,
                    confidence=1.0,
                    inference_ms=1.0,
                )
            )
            tgt.append((t.nx, t.ny))
    model = RidgeCalibration(alpha=0.01)
    model.fit(raw, tgt)
    for t in targets:
        r = RawGaze(x=t.nx * 1.05 + 0.02, y=t.ny * 0.95 - 0.02, confidence=1.0, inference_ms=1.0)
        out = model.transform(r)
        assert out.x == pytest.approx(t.nx, abs=0.03)
        assert out.y == pytest.approx(t.ny, abs=0.03)


def test_ridge_rejects_negative_alpha() -> None:
    with pytest.raises(CalibrationError, match="alpha"):
        RidgeCalibration(alpha=-0.1)


# ---------------------------------------------------------------------------
# Round-trip through to_dict / from_dict (the "reload numerically identical" DoD)
# ---------------------------------------------------------------------------
def test_affine_serialisation_round_trip_is_bit_identical() -> None:
    raw, tgt = _samples_with_affine_bias()
    model = AffineCalibration(profile_id="alice")
    model.fit(raw, tgt)
    reloaded = AffineCalibration.from_dict(model.to_dict())
    assert reloaded.profile_id == model.profile_id
    assert model.weights is not None
    assert reloaded.weights is not None
    np.testing.assert_array_equal(reloaded.weights, model.weights)
    sample = RawGaze(x=0.42, y=0.71, confidence=1.0, inference_ms=1.0)
    assert model.transform(sample) == reloaded.transform(sample)


def test_polynomial_serialisation_round_trip_is_bit_identical() -> None:
    raw, tgt = _samples_with_affine_bias(scale_x=1.2, scale_y=0.85)
    model = PolynomialCalibration(profile_id="bob")
    model.fit(raw, tgt)
    reloaded = PolynomialCalibration.from_dict(model.to_dict())
    assert model.weights is not None
    assert reloaded.weights is not None
    np.testing.assert_array_equal(reloaded.weights, model.weights)


def test_ridge_serialisation_round_trip_is_bit_identical() -> None:
    raw, tgt = _samples_with_affine_bias()
    model = RidgeCalibration(profile_id="carol", alpha=0.005)
    model.fit(raw, tgt)
    reloaded = RidgeCalibration.from_dict(model.to_dict())
    assert reloaded.alpha == model.alpha
    assert model.weights is not None
    assert reloaded.weights is not None
    np.testing.assert_array_equal(reloaded.weights, model.weights)


def test_unfitted_model_cannot_serialise() -> None:
    with pytest.raises(CalibrationError, match="unfitted"):
        AffineCalibration().to_dict()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_dispatches_by_kind() -> None:
    assert isinstance(build_calibration_model("affine"), AffineCalibration)
    assert isinstance(build_calibration_model("polynomial"), PolynomialCalibration)
    assert isinstance(build_calibration_model("ridge"), RidgeCalibration)


def test_factory_rejects_unknown_kind() -> None:
    with pytest.raises(CalibrationError, match="unknown calibration"):
        build_calibration_model("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# from_dict validation
# ---------------------------------------------------------------------------
def test_affine_from_dict_rejects_wrong_kind() -> None:
    with pytest.raises(CalibrationError, match="from_dict"):
        AffineCalibration.from_dict({"kind": "polynomial", "weights": [[0, 0]] * 3})


def test_affine_from_dict_rejects_wrong_shape() -> None:
    with pytest.raises(CalibrationError, match="weights shape"):
        AffineCalibration.from_dict({"kind": "affine", "weights": [[0, 0]] * 4})


def test_polynomial_from_dict_rejects_wrong_kind() -> None:
    with pytest.raises(CalibrationError, match="from_dict"):
        PolynomialCalibration.from_dict({"kind": "affine", "weights": [[0, 0]] * 6})


def test_polynomial_from_dict_rejects_wrong_shape() -> None:
    with pytest.raises(CalibrationError, match="weights shape"):
        PolynomialCalibration.from_dict({"kind": "polynomial", "weights": [[0, 0]] * 3})


def test_ridge_from_dict_rejects_wrong_kind() -> None:
    with pytest.raises(CalibrationError, match="from_dict"):
        RidgeCalibration.from_dict({"kind": "affine", "weights": [[0, 0]] * 6})


def test_ridge_from_dict_rejects_wrong_shape() -> None:
    with pytest.raises(CalibrationError, match="weights shape"):
        RidgeCalibration.from_dict({"kind": "ridge", "weights": [[0, 0]] * 3})


# ---------------------------------------------------------------------------
# is_fitted flag + polynomial/ridge unfitted transform paths
# ---------------------------------------------------------------------------
def test_polynomial_transform_before_fit_raises() -> None:
    with pytest.raises(CalibrationError, match="before fit"):
        PolynomialCalibration().transform(RawGaze(x=0.5, y=0.5, confidence=1.0, inference_ms=1.0))


def test_ridge_transform_before_fit_raises() -> None:
    with pytest.raises(CalibrationError, match="before fit"):
        RidgeCalibration().transform(RawGaze(x=0.5, y=0.5, confidence=1.0, inference_ms=1.0))


def test_polynomial_is_fitted_flag_flip() -> None:
    model = PolynomialCalibration()
    assert model.is_fitted is False
    raw, tgt = _samples_with_affine_bias(per_target=8)
    model.fit(raw, tgt)
    assert model.is_fitted is True


def test_ridge_is_fitted_flag_flip() -> None:
    model = RidgeCalibration()
    assert model.is_fitted is False
    raw, tgt = _samples_with_affine_bias(per_target=8)
    model.fit(raw, tgt)
    assert model.is_fitted is True


def test_polynomial_and_ridge_to_dict_rejects_unfitted() -> None:
    with pytest.raises(CalibrationError, match="unfitted"):
        PolynomialCalibration().to_dict()
    with pytest.raises(CalibrationError, match="unfitted"):
        RidgeCalibration().to_dict()


def test_ridge_underdetermined_raises() -> None:
    with pytest.raises(CalibrationError, match="need at least"):
        RidgeCalibration().fit(
            [RawGaze(x=0.1, y=0.1, confidence=1.0, inference_ms=1.0)] * 2,
            [(0.5, 0.5), (0.6, 0.6)],
        )
