"""Sprint 8 -- quality metrics and hold-out split."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.calibration.models import AffineCalibration, PolynomialCalibration
from irisflow.calibration.points import generate_targets
from irisflow.calibration.quality import (
    evaluate_calibration,
    hold_out_split,
    identify_bad_targets,
)
from irisflow.core.exceptions import CalibrationError
from irisflow.core.types import RawGaze


def _biased_dataset(
    scale_x: float = 1.15,
    scale_y: float = 0.85,
    offset_x: float = 0.03,
    offset_y: float = -0.04,
    per_target: int = 30,
    seed: int = 0,
) -> tuple[list[RawGaze], list[tuple[float, float]], list[int]]:
    rng = np.random.default_rng(seed)
    targets = generate_targets(9, margin=0.1)
    raw: list[RawGaze] = []
    tgt: list[tuple[float, float]] = []
    indices: list[int] = []
    for t in targets:
        for _ in range(per_target):
            noise = float(rng.normal(0.0, 0.003))
            raw.append(
                RawGaze(
                    x=t.nx * scale_x + offset_x + noise,
                    y=t.ny * scale_y + offset_y + noise,
                    confidence=1.0,
                    inference_ms=1.0,
                )
            )
            tgt.append((t.nx, t.ny))
            indices.append(t.index)
    return raw, tgt, indices


class _PassthroughModel:
    """Naive calibration -- returns raw values as-is."""

    def fit(self, raw_samples, targets) -> None:  # type: ignore[no-untyped-def]
        return None

    def transform(self, raw: RawGaze):  # type: ignore[no-untyped-def]
        from irisflow.core.types import CalibratedGaze

        return CalibratedGaze(x=raw.x, y=raw.y, profile_id="passthrough")

    @property
    def is_fitted(self) -> bool:
        return True


def test_evaluate_returns_error_in_pixels() -> None:
    raw, tgt, indices = _biased_dataset()
    model = _PassthroughModel()
    report = evaluate_calibration(
        model,
        raw,
        tgt,
        screen_width=1920,
        screen_height=1080,
        target_indices=indices,
    )
    # Under our bias (scale ~1.15) the pixel error should be substantial.
    assert report.mean_error_px > 20.0
    assert report.p95_error_px >= report.mean_error_px
    assert report.max_error_px >= report.p95_error_px
    assert len(report.targets) == 9
    # per-target list is sorted worst first
    errors = [t.error_px for t in report.targets]
    assert errors == sorted(errors, reverse=True)


def test_polynomial_reduces_error_at_least_30pct_vs_passthrough() -> None:
    """This is the numeric DoD criterion for Sprint 8."""
    raw, tgt, indices = _biased_dataset(per_target=40, seed=1)
    train_idx, holdout_idx = hold_out_split(raw, tgt, indices, hold_out_fraction=0.25, seed=1)

    train_raw = [raw[i] for i in train_idx]
    train_tgt = [tgt[i] for i in train_idx]
    holdout_raw = [raw[i] for i in holdout_idx]
    holdout_tgt = [tgt[i] for i in holdout_idx]
    holdout_indices = [indices[i] for i in holdout_idx]

    passthrough = _PassthroughModel()
    poly = PolynomialCalibration()
    poly.fit(train_raw, train_tgt)

    baseline = evaluate_calibration(
        passthrough,
        holdout_raw,
        holdout_tgt,
        screen_width=1920,
        screen_height=1080,
        target_indices=holdout_indices,
    )
    calibrated = evaluate_calibration(
        poly,
        holdout_raw,
        holdout_tgt,
        screen_width=1920,
        screen_height=1080,
        target_indices=holdout_indices,
    )
    reduction = 1.0 - calibrated.mean_error_px / baseline.mean_error_px
    assert reduction >= 0.30, (
        f"expected >=30% error reduction, got {reduction * 100:.1f}%; "
        f"baseline={baseline.mean_error_px:.1f}px calibrated={calibrated.mean_error_px:.1f}px"
    )


def test_hold_out_split_is_stratified_per_target() -> None:
    raw, tgt, indices = _biased_dataset(per_target=10)
    train, holdout = hold_out_split(raw, tgt, indices, hold_out_fraction=0.2, seed=0)
    assert set(train).isdisjoint(set(holdout))
    assert len(train) + len(holdout) == len(raw)
    # Every target must appear in both splits.
    train_targets = {indices[i] for i in train}
    holdout_targets = {indices[i] for i in holdout}
    assert train_targets == set(indices)
    assert holdout_targets == set(indices)


def test_hold_out_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="hold_out_fraction"):
        hold_out_split([], [], [], hold_out_fraction=0.0)


def test_evaluate_rejects_mismatched_shapes() -> None:
    with pytest.raises(CalibrationError):
        evaluate_calibration(
            _PassthroughModel(),
            [RawGaze(x=0.5, y=0.5, confidence=1.0, inference_ms=1.0)],
            [(0.1, 0.1), (0.2, 0.2)],
            screen_width=1920,
            screen_height=1080,
        )


def test_identify_bad_targets_picks_worst() -> None:
    raw, tgt, indices = _biased_dataset(per_target=20)
    # Inject one outlier target with a large offset
    for i, idx in enumerate(indices):
        if idx == 0:
            raw[i] = RawGaze(x=raw[i].x + 0.3, y=raw[i].y, confidence=1.0, inference_ms=1.0)
    model = AffineCalibration()
    model.fit(raw, tgt)
    report = evaluate_calibration(
        model,
        raw,
        tgt,
        screen_width=1920,
        screen_height=1080,
        target_indices=indices,
    )
    bad = identify_bad_targets(report, threshold_px=50.0)
    assert 0 in bad
