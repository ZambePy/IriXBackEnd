"""Calibration transforms — raw normalized gaze → calibrated normalized gaze.

Three implementations, one Protocol (:class:`~irisflow.core.interfaces.CalibrationModel`):

* :class:`AffineCalibration` — 6 parameters (linear + bias), 2D→2D.
  Cheap, robust, cannot correct barrel distortion.
* :class:`PolynomialCalibration` — second-order features
  ``[1, x, y, x², xy, y²]`` per output, 12 parameters. The SPRINTS
  default: corrects the mild bowing you see when the head is tilted or
  the webcam is off-center.
* :class:`RidgeCalibration` — same feature basis as polynomial with L2
  regularisation, safer when the user rushed the calibration and some
  targets have very few samples.

Every implementation solves the normal equations by least squares —
``numpy.linalg.lstsq`` is fine for problems this small (<= 13 targets
by 30 samples = 390 rows, tiny). Rejecting NaN/inf on inputs is on the
caller (:mod:`irisflow.calibration.collector` already filters those out).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irisflow.core.exceptions import CalibrationError
from irisflow.core.types import CalibratedGaze, RawGaze

__all__ = [
    "AffineCalibration",
    "CalibrationKind",
    "PolynomialCalibration",
    "RidgeCalibration",
    "build_calibration_model",
]


CalibrationKind = Literal["affine", "polynomial", "ridge"]


# ---------------------------------------------------------------------------
# Feature bases — shared by every model so the fit/transform code stays tiny.
# ---------------------------------------------------------------------------
def _affine_features(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the ``(N, 3)`` design matrix ``[1, x, y]``."""
    ones = np.ones_like(xs)
    return np.stack([ones, xs, ys], axis=1)


def _polynomial_features(
    xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the ``(N, 6)`` design matrix ``[1, x, y, x², xy, y²]``."""
    ones = np.ones_like(xs)
    return np.stack([ones, xs, ys, xs * xs, xs * ys, ys * ys], axis=1)


def _prepare_samples(
    raw_samples: Sequence[RawGaze],
    targets: Sequence[tuple[float, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and stack ``(raw_samples, targets)`` into aligned arrays."""
    if len(raw_samples) != len(targets):
        raise CalibrationError(
            f"fit: {len(raw_samples)} raw samples vs {len(targets)} targets "
            "— they must be aligned pairwise"
        )
    if len(raw_samples) == 0:
        raise CalibrationError("fit: at least one (raw, target) pair is required")

    raw = np.asarray([(s.x, s.y) for s in raw_samples], dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(tgt).all():
        raise CalibrationError("fit: samples/targets contain NaN or inf")
    return raw, tgt


def _least_squares(
    design: NDArray[np.float64], targets: NDArray[np.float64], *, min_rows: int
) -> NDArray[np.float64]:
    """Solve ``design @ W = targets`` for ``W`` via ``lstsq``.

    Raises :class:`CalibrationError` when the problem is under-determined
    (``design`` has fewer rows than parameters). This is the fast-fail
    that keeps a mis-configured calibration session from producing a
    silently useless model.
    """
    n_rows, n_cols = design.shape
    if n_rows < max(min_rows, n_cols):
        raise CalibrationError(
            f"fit: only {n_rows} samples for a {n_cols}-parameter model; "
            f"need at least {max(min_rows, n_cols)}"
        )
    weights, _residuals, _rank, _sv = np.linalg.lstsq(design, targets, rcond=None)
    if not np.isfinite(weights).all():
        raise CalibrationError("fit: solver produced NaN/inf weights")
    return weights.astype(np.float64, copy=False)  # shape (n_cols, 2)


def _ridge(
    design: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    alpha: float,
    min_rows: int,
) -> NDArray[np.float64]:
    """Solve the ridge-regularised normal equations directly.

    ``(X.T @ X + alpha*I) W = X.T @ y`` -- closed form and stable for the small
    ``(N <= 400, k <= 6)`` problems we hand it. The intercept column
    (index 0) is exempt from the penalty -- a bias correction should not
    be shrunk toward zero.
    """
    n_rows, n_cols = design.shape
    if n_rows < min_rows:
        raise CalibrationError(
            f"fit: only {n_rows} samples for ridge; need at least {min_rows}"
        )
    gram = design.T @ design
    penalty = alpha * np.eye(n_cols, dtype=np.float64)
    penalty[0, 0] = 0.0
    try:
        weights = np.linalg.solve(gram + penalty, design.T @ targets)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError(f"fit: ridge solver failed ({exc})") from exc
    if not np.isfinite(weights).all():
        raise CalibrationError("fit: ridge solver produced NaN/inf weights")
    return weights.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------
@dataclass
class AffineCalibration:
    """Six-parameter affine correction ``(x, y) → A·(x, y) + b``."""

    profile_id: str = "default"
    weights: NDArray[np.float64] | None = None

    kind: str = field(init=False, default="affine")

    def fit(
        self,
        raw_samples: Sequence[RawGaze],
        targets: Sequence[tuple[float, float]],
    ) -> None:
        raw, tgt = _prepare_samples(raw_samples, targets)
        design = _affine_features(raw[:, 0], raw[:, 1])
        self.weights = _least_squares(design, tgt, min_rows=3)

    def transform(self, raw: RawGaze) -> CalibratedGaze:
        if self.weights is None:
            raise CalibrationError("AffineCalibration.transform called before fit")
        design = _affine_features(
            np.asarray([raw.x], dtype=np.float64),
            np.asarray([raw.y], dtype=np.float64),
        )
        out = (design @ self.weights).flatten()
        return CalibratedGaze(x=float(out[0]), y=float(out[1]), profile_id=self.profile_id)

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    def to_dict(self) -> dict[str, Any]:
        if self.weights is None:
            raise CalibrationError("cannot serialise an unfitted AffineCalibration")
        return {
            "kind": self.kind,
            "profile_id": self.profile_id,
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AffineCalibration:
        if data.get("kind") != "affine":
            raise CalibrationError(
                f"AffineCalibration.from_dict: kind={data.get('kind')!r}"
            )
        weights = np.asarray(data["weights"], dtype=np.float64)
        if weights.shape != (3, 2):
            raise CalibrationError(
                f"AffineCalibration.from_dict: weights shape {weights.shape} != (3, 2)"
            )
        return cls(profile_id=str(data.get("profile_id", "default")), weights=weights)


@dataclass
class PolynomialCalibration:
    """Second-order polynomial correction — the SPRINTS default (§8)."""

    profile_id: str = "default"
    degree: int = 2
    weights: NDArray[np.float64] | None = None

    kind: str = field(init=False, default="polynomial")

    def __post_init__(self) -> None:
        if self.degree != 2:
            raise CalibrationError(
                f"PolynomialCalibration only supports degree=2, got {self.degree}"
            )

    def fit(
        self,
        raw_samples: Sequence[RawGaze],
        targets: Sequence[tuple[float, float]],
    ) -> None:
        raw, tgt = _prepare_samples(raw_samples, targets)
        design = _polynomial_features(raw[:, 0], raw[:, 1])
        self.weights = _least_squares(design, tgt, min_rows=6)

    def transform(self, raw: RawGaze) -> CalibratedGaze:
        if self.weights is None:
            raise CalibrationError("PolynomialCalibration.transform called before fit")
        design = _polynomial_features(
            np.asarray([raw.x], dtype=np.float64),
            np.asarray([raw.y], dtype=np.float64),
        )
        out = (design @ self.weights).flatten()
        return CalibratedGaze(x=float(out[0]), y=float(out[1]), profile_id=self.profile_id)

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    def to_dict(self) -> dict[str, Any]:
        if self.weights is None:
            raise CalibrationError("cannot serialise an unfitted PolynomialCalibration")
        return {
            "kind": self.kind,
            "profile_id": self.profile_id,
            "degree": self.degree,
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolynomialCalibration:
        if data.get("kind") != "polynomial":
            raise CalibrationError(
                f"PolynomialCalibration.from_dict: kind={data.get('kind')!r}"
            )
        weights = np.asarray(data["weights"], dtype=np.float64)
        if weights.shape != (6, 2):
            raise CalibrationError(
                f"PolynomialCalibration.from_dict: weights shape {weights.shape} != (6, 2)"
            )
        return cls(
            profile_id=str(data.get("profile_id", "default")),
            degree=int(data.get("degree", 2)),
            weights=weights,
        )


@dataclass
class RidgeCalibration:
    """L2-regularised polynomial calibration — safer under scarce samples.

    ``alpha`` is the regularisation strength; larger values pull the
    correction toward the identity, protecting against overfitting to
    noisy per-target samples. ``0.01`` works well for the 9-point/
    30-sample regime; halve it for 13 points, double it for 5.
    """

    profile_id: str = "default"
    alpha: float = 0.01
    weights: NDArray[np.float64] | None = None

    kind: str = field(init=False, default="ridge")

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise CalibrationError(f"ridge alpha must be >= 0, got {self.alpha}")

    def fit(
        self,
        raw_samples: Sequence[RawGaze],
        targets: Sequence[tuple[float, float]],
    ) -> None:
        raw, tgt = _prepare_samples(raw_samples, targets)
        design = _polynomial_features(raw[:, 0], raw[:, 1])
        self.weights = _ridge(design, tgt, alpha=self.alpha, min_rows=3)

    def transform(self, raw: RawGaze) -> CalibratedGaze:
        if self.weights is None:
            raise CalibrationError("RidgeCalibration.transform called before fit")
        design = _polynomial_features(
            np.asarray([raw.x], dtype=np.float64),
            np.asarray([raw.y], dtype=np.float64),
        )
        out = (design @ self.weights).flatten()
        return CalibratedGaze(x=float(out[0]), y=float(out[1]), profile_id=self.profile_id)

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    def to_dict(self) -> dict[str, Any]:
        if self.weights is None:
            raise CalibrationError("cannot serialise an unfitted RidgeCalibration")
        return {
            "kind": self.kind,
            "profile_id": self.profile_id,
            "alpha": self.alpha,
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RidgeCalibration:
        if data.get("kind") != "ridge":
            raise CalibrationError(
                f"RidgeCalibration.from_dict: kind={data.get('kind')!r}"
            )
        weights = np.asarray(data["weights"], dtype=np.float64)
        if weights.shape != (6, 2):
            raise CalibrationError(
                f"RidgeCalibration.from_dict: weights shape {weights.shape} != (6, 2)"
            )
        return cls(
            profile_id=str(data.get("profile_id", "default")),
            alpha=float(data.get("alpha", 0.01)),
            weights=weights,
        )


# ---------------------------------------------------------------------------
# Factory — kept here so callers can dispatch by config kind without
# importing three different classes.
# ---------------------------------------------------------------------------
def build_calibration_model(
    kind: CalibrationKind,
    *,
    profile_id: str = "default",
    polynomial_degree: int = 2,
    ridge_alpha: float = 0.01,
) -> AffineCalibration | PolynomialCalibration | RidgeCalibration:
    """Construct the calibration model referenced by config."""
    if kind == "affine":
        return AffineCalibration(profile_id=profile_id)
    if kind == "polynomial":
        return PolynomialCalibration(profile_id=profile_id, degree=polynomial_degree)
    if kind == "ridge":
        return RidgeCalibration(profile_id=profile_id, alpha=ridge_alpha)
    raise CalibrationError(f"unknown calibration kind {kind!r}")
