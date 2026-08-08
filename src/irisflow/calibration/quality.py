"""Post-fit quality metrics: residual error, hold-out validation, bad-target ID.

The DoD for Sprint 8 requires that we (a) know how good a fit is,
(b) know which targets contributed most to the error so the session can
offer partial re-collection, and (c) express both in screen pixels so
the number is meaningful to the operator.

All functions are pure and operate on numpy arrays plus the calibrated
:class:`~irisflow.core.interfaces.CalibrationModel`. No I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from irisflow.core.exceptions import CalibrationError
from irisflow.core.interfaces import CalibrationModel
from irisflow.core.types import RawGaze

__all__ = [
    "CalibrationQualityReport",
    "TargetError",
    "evaluate_calibration",
    "hold_out_split",
    "identify_bad_targets",
]


@dataclass(frozen=True, slots=True)
class TargetError:
    """Per-target residual, sorted from best to worst by :func:`evaluate_calibration`."""

    target_index: int
    target_nx: float
    target_ny: float
    predicted_nx: float
    predicted_ny: float
    error_px: float


@dataclass(frozen=True, slots=True)
class CalibrationQualityReport:
    """Summary of one calibrated model's performance on a dataset.

    ``mean_error_px`` and ``max_error_px`` are the numbers to show the
    user. ``targets`` holds the per-target breakdown so the session can
    ask for a partial re-collection of the worst ones.
    """

    mean_error_px: float
    max_error_px: float
    p95_error_px: float
    screen_width: int
    screen_height: int
    n_samples: int
    targets: list[TargetError] = field(default_factory=list)


def evaluate_calibration(
    model: CalibrationModel,
    raw_samples: Sequence[RawGaze],
    targets: Sequence[tuple[float, float]],
    *,
    screen_width: int,
    screen_height: int,
    target_indices: Sequence[int] | None = None,
) -> CalibrationQualityReport:
    """Score ``model`` against ``(raw_samples, targets)`` pairs in **pixel** units.

    The pixel conversion uses the diagonal-preserving mapping
    ``dx² · Wpx² + dy² · Hpx²`` so an equal-normalized deviation on a
    wide screen costs more pixels horizontally than vertically — this is
    what actually matters to a user clicking a button.

    Args:
        target_indices: Optional index (into ``targets``) each sample
            belongs to. When provided, the returned report groups errors
            by target; otherwise all samples land in a single implicit
            group.
    """
    if len(raw_samples) != len(targets):
        raise CalibrationError(f"evaluate: {len(raw_samples)} samples vs {len(targets)} targets")
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError(f"screen dims must be positive, got {screen_width}x{screen_height}")
    if len(raw_samples) == 0:
        raise CalibrationError("evaluate: no samples provided")
    if target_indices is not None and len(target_indices) != len(raw_samples):
        raise CalibrationError(
            f"evaluate: target_indices length ({len(target_indices)}) "
            f"!= samples length ({len(raw_samples)})"
        )

    predictions = [model.transform(s) for s in raw_samples]
    pred_arr = np.asarray([(p.x, p.y) for p in predictions], dtype=np.float64)
    tgt_arr = np.asarray(targets, dtype=np.float64)

    delta = pred_arr - tgt_arr
    errors_px = _pixel_error(delta, screen_width, screen_height)

    mean_px = float(errors_px.mean())
    max_px = float(errors_px.max())
    p95_px = float(np.percentile(errors_px, 95))

    per_target: list[TargetError] = []
    if target_indices is not None:
        indices_arr = np.asarray(target_indices, dtype=np.int64)
        for idx in np.unique(indices_arr):
            mask = indices_arr == idx
            avg_pred = pred_arr[mask].mean(axis=0)
            avg_target = tgt_arr[mask].mean(axis=0)
            per_target.append(
                TargetError(
                    target_index=int(idx),
                    target_nx=float(avg_target[0]),
                    target_ny=float(avg_target[1]),
                    predicted_nx=float(avg_pred[0]),
                    predicted_ny=float(avg_pred[1]),
                    error_px=float(errors_px[mask].mean()),
                )
            )
        per_target.sort(key=lambda t: t.error_px, reverse=True)

    return CalibrationQualityReport(
        mean_error_px=mean_px,
        max_error_px=max_px,
        p95_error_px=p95_px,
        screen_width=int(screen_width),
        screen_height=int(screen_height),
        n_samples=len(raw_samples),
        targets=per_target,
    )


def _pixel_error(
    delta: NDArray[np.float64], screen_width: int, screen_height: int
) -> NDArray[np.float64]:
    """Euclidean pixel error from a normalized delta ``(dx, dy)`` per row."""
    dxp = delta[:, 0] * float(screen_width)
    dyp = delta[:, 1] * float(screen_height)
    return np.sqrt(dxp * dxp + dyp * dyp)


def hold_out_split(
    raw_samples: Sequence[RawGaze],
    targets: Sequence[tuple[float, float]],
    target_indices: Sequence[int],
    *,
    hold_out_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Stratified train/hold-out split by target index.

    Every target gets ``ceil(N * hold_out_fraction)`` of its samples set
    aside for evaluation — this is what the DoD "polynomial reduces error
    ≥ 30% vs passthrough" criterion is measured on.

    Returns two lists of **sample indices** (into the original arrays):
    ``(train_indices, hold_out_indices)``.
    """
    if not 0.0 < hold_out_fraction < 1.0:
        raise ValueError(f"hold_out_fraction must be in (0, 1), got {hold_out_fraction}")
    if len(raw_samples) != len(targets) or len(raw_samples) != len(target_indices):
        raise CalibrationError("hold_out_split: samples/targets/target_indices must be aligned")

    rng = np.random.default_rng(seed)
    indices_arr = np.asarray(target_indices, dtype=np.int64)
    train: list[int] = []
    holdout: list[int] = []
    for idx in np.unique(indices_arr):
        rows = np.where(indices_arr == idx)[0]
        rng.shuffle(rows)
        n_hold = max(1, int(np.ceil(len(rows) * hold_out_fraction)))
        holdout.extend(int(i) for i in rows[:n_hold])
        train.extend(int(i) for i in rows[n_hold:])
    if not train:
        raise CalibrationError(
            "hold_out_split: not enough samples per target to leave any for training"
        )
    return train, holdout


def identify_bad_targets(report: CalibrationQualityReport, *, threshold_px: float) -> list[int]:
    """Return indices of targets whose mean error exceeds ``threshold_px``.

    Sorted worst-first so a UI can offer them for recollection in that
    order without another sort step.
    """
    if threshold_px <= 0:
        raise ValueError(f"threshold_px must be > 0, got {threshold_px}")
    return [t.target_index for t in report.targets if t.error_px > threshold_px]
