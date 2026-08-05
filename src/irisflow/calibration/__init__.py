"""Calibration layer (Sprint 8).

Public API:

* :class:`~irisflow.calibration.points.CalibrationTarget` /
  :func:`~irisflow.calibration.points.generate_targets` — screen grids.
* :class:`~irisflow.calibration.collector.TargetCollector` — accepts
  :class:`~irisflow.core.types.RawGaze` frames one at a time.
* :class:`~irisflow.calibration.models.AffineCalibration`,
  :class:`~irisflow.calibration.models.PolynomialCalibration`,
  :class:`~irisflow.calibration.models.RidgeCalibration` and the
  :func:`~irisflow.calibration.models.build_calibration_model` factory.
* :func:`~irisflow.calibration.quality.evaluate_calibration` and
  :func:`~irisflow.calibration.quality.hold_out_split` for scoring.
* :class:`~irisflow.calibration.session.CalibrationSession` — the
  end-to-end state machine.
* :class:`~irisflow.calibration.store.CalibrationStore` /
  :class:`~irisflow.calibration.store.CalibrationProfile` — JSON
  persistence.
"""

from irisflow.calibration.collector import (
    CollectorState,
    TargetCollector,
    TargetSampleBatch,
    flatten_batches,
)
from irisflow.calibration.models import (
    AffineCalibration,
    CalibrationKind,
    PolynomialCalibration,
    RidgeCalibration,
    build_calibration_model,
)
from irisflow.calibration.points import (
    CalibrationTarget,
    TargetCount,
    generate_targets,
)
from irisflow.calibration.quality import (
    CalibrationQualityReport,
    TargetError,
    evaluate_calibration,
    hold_out_split,
    identify_bad_targets,
)
from irisflow.calibration.session import (
    CalibrationOutcome,
    CalibrationSession,
    SessionPhase,
)
from irisflow.calibration.store import (
    CalibrationProfile,
    CalibrationStore,
    build_profile_from_report,
    load_profile_dict,
    save_profile_dict,
)

__all__ = [
    "AffineCalibration",
    "CalibrationKind",
    "CalibrationOutcome",
    "CalibrationProfile",
    "CalibrationQualityReport",
    "CalibrationSession",
    "CalibrationStore",
    "CalibrationTarget",
    "CollectorState",
    "PolynomialCalibration",
    "RidgeCalibration",
    "SessionPhase",
    "TargetCollector",
    "TargetCount",
    "TargetError",
    "TargetSampleBatch",
    "build_calibration_model",
    "build_profile_from_report",
    "evaluate_calibration",
    "flatten_batches",
    "generate_targets",
    "hold_out_split",
    "identify_bad_targets",
    "load_profile_dict",
    "save_profile_dict",
]
