"""State machine that drives one calibration session end-to-end.

Composes the pieces built in Sprint 8:

* :func:`~irisflow.calibration.points.generate_targets` — the grid.
* :class:`~irisflow.calibration.collector.TargetCollector` — one per
  target, gathering :class:`~irisflow.core.types.RawGaze` samples.
* :class:`~irisflow.calibration.models.CalibrationModel` — fitted from
  the pooled samples once every target reports ``CollectorState.DONE``.
* :func:`~irisflow.calibration.quality.evaluate_calibration` — computes
  the residual error the session reports back.
* :class:`~irisflow.calibration.store.CalibrationProfile` — the on-disk
  format the fitted model is packaged into.

The session is layer-neutral: it doesn't know about
:mod:`irisflow.pipeline.bus` — the caller passes in a ``publisher``
callable (usually the bus' ``publish``). This keeps
:mod:`irisflow.calibration` free of any inbound dependency on the
pipeline layer, which ``import-linter`` would reject.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from irisflow.calibration.collector import (
    CollectorState,
    TargetCollector,
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
    evaluate_calibration,
    hold_out_split,
    identify_bad_targets,
)
from irisflow.calibration.store import (
    CalibrationProfile,
    build_profile_from_report,
)
from irisflow.core.clock import Clock, SystemClock
from irisflow.core.events import CalibrationPhase, CalibrationProgress
from irisflow.core.exceptions import CalibrationError
from irisflow.core.types import RawGaze

__all__ = [
    "CalibrationOutcome",
    "CalibrationSession",
    "SessionPhase",
]


ProgressPublisher = Callable[[CalibrationProgress], None]


class SessionPhase(StrEnum):
    """High-level phase of the session — coarser than per-target state."""

    IDLE = "idle"
    COLLECTING = "collecting"
    FITTING = "fitting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    """Result returned by :meth:`CalibrationSession.finalize`.

    ``accepted`` is true when the fit met the ``max_residual_px``
    threshold from :class:`~irisflow.config.schema.CalibrationConfig`;
    when false, ``bad_targets`` lists the target indices worth retrying.
    """

    profile: CalibrationProfile
    report: CalibrationQualityReport
    holdout_report: CalibrationQualityReport
    accepted: bool
    bad_targets: list[int] = field(default_factory=list)


@dataclass
class CalibrationSession:
    """Drive one calibration end-to-end (§4.2 in SPRINTS.md).

    Args:
        profile_id: Name used when persisting the resulting profile.
        screen_width, screen_height: Pixel dimensions of the target
            screen — required so quality metrics come out in pixels.
        point_count: 5, 9 or 13 (§8 default is 9).
        samples_per_point, stabilization_ms, model_kind, max_residual_px,
            polynomial_degree, ridge_alpha: Mirror the fields of
            :class:`~irisflow.config.schema.CalibrationConfig`. Passing
            primitives (not the pydantic model) keeps
            :mod:`irisflow.calibration` free of a dependency on
            :mod:`irisflow.config`, which is a sibling layer.
        margin: Passed through to :func:`generate_targets`.
        publisher: Optional callback fired on every :class:`CalibrationProgress`
            event. In production the caller passes ``bus.publish``.
        clock: Injected for deterministic tests.
    """

    profile_id: str
    screen_width: int
    screen_height: int
    point_count: TargetCount = 9
    samples_per_point: int = 30
    stabilization_ms: int = 600
    model_kind: CalibrationKind = "polynomial"
    max_residual_px: float = 60.0
    polynomial_degree: int = 2
    ridge_alpha: float = 0.01
    margin: float = 0.08
    publisher: ProgressPublisher | None = None
    clock: Clock = field(default_factory=SystemClock)
    hold_out_fraction: float = 0.2
    hold_out_seed: int = 0

    _phase: SessionPhase = field(default=SessionPhase.IDLE, init=False)
    _targets: list[CalibrationTarget] = field(default_factory=list, init=False)
    _collectors: dict[int, TargetCollector] = field(default_factory=dict, init=False)
    _active_index: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise CalibrationError("profile_id must be non-empty")
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise CalibrationError(
                f"screen must have positive dims, got {self.screen_width}x{self.screen_height}"
            )
        if self.max_residual_px <= 0:
            raise CalibrationError(
                f"max_residual_px must be > 0, got {self.max_residual_px}"
            )
        self._targets = generate_targets(self.point_count, margin=self.margin)

    # ------------------------------------------------------------------ props
    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def targets(self) -> list[CalibrationTarget]:
        return list(self._targets)

    @property
    def total_targets(self) -> int:
        return len(self._targets)

    @property
    def active_target(self) -> CalibrationTarget | None:
        if self._active_index is None:
            return None
        return self._targets[self._active_index]

    @property
    def is_current_target_done(self) -> bool:
        idx = self._active_index
        if idx is None:
            return False
        collector = self._collectors.get(idx)
        return collector is not None and collector.is_done

    @property
    def current_target_collected(self) -> int:
        """Number of samples accepted so far for the active target."""
        idx = self._active_index
        if idx is None:
            return 0
        collector = self._collectors.get(idx)
        return 0 if collector is None else collector.collected

    def force_complete_active(self) -> None:
        """Mark the active target as done with the samples collected so far.

        Called by drivers on a wall-clock budget; raises if no samples
        have been accepted yet (that path should call :meth:`abort_target`).
        """
        idx = self._active_index
        if idx is None:
            raise CalibrationError("force_complete_active called without an active target")
        collector = self._collectors[idx]
        collector.force_done()
        self._emit(self._targets[idx], phase="settled")

    @property
    def is_collection_complete(self) -> bool:
        """True once every target has been resolved (done or explicitly aborted)."""
        if not self._collectors:
            return False
        for t in self._targets:
            collector = self._collectors.get(t.index)
            if collector is None:
                return False
            if collector.state not in (CollectorState.DONE, CollectorState.ABORTED):
                return False
        return True

    # ------------------------------------------------------------------ workflow
    def start(self) -> None:
        """Move from IDLE to COLLECTING and reset any prior state."""
        self._phase = SessionPhase.COLLECTING
        self._collectors = {}
        self._active_index = None

    def begin_target(self, index: int) -> CalibrationTarget:
        """Announce that the UI is now showing target ``index``.

        Creates or replaces the collector for that target and publishes a
        :class:`CalibrationProgress` event with phase ``"prompt"``. The
        clock reading at this moment defines the ``stabilization_ms``
        window every :meth:`push_sample` compares against.
        """
        if self._phase not in (SessionPhase.COLLECTING, SessionPhase.IDLE):
            raise CalibrationError(
                f"begin_target requires COLLECTING/IDLE, got {self._phase.value}"
            )
        if self._phase == SessionPhase.IDLE:
            self.start()
        target = self._require_target(index)
        collector = TargetCollector(
            target_index=target.index,
            target_nx=target.nx,
            target_ny=target.ny,
            samples_per_point=self.samples_per_point,
            stabilization_ms=self.stabilization_ms,
        )
        collector.mark_shown(self.clock.monotonic())
        self._collectors[target.index] = collector
        self._active_index = target.index
        self._emit(target, phase="prompt")
        return target

    def push_sample(self, raw: RawGaze | None, timestamp_s: float) -> None:
        """Feed one frame's raw gaze (``None`` = face lost) to the active target."""
        if self._active_index is None:
            raise CalibrationError("push_sample called before begin_target")
        collector = self._collectors[self._active_index]
        prev_state = collector.state
        collector.push_sample(raw, timestamp_s)
        target = self._targets[self._active_index]
        if prev_state != collector.state:
            phase: CalibrationPhase
            if collector.state == CollectorState.COLLECTING:
                phase = "collecting"
            elif collector.state == CollectorState.DONE:
                phase = "settled"
            elif collector.state == CollectorState.ABORTED:
                phase = "aborted"
            else:
                phase = "prompt"
            self._emit(target, phase=phase)

    def abort_target(self, index: int) -> None:
        """Give up on a specific target — session can call :meth:`begin_target`
        again later to retry it."""
        collector = self._collectors.get(index)
        if collector is None:
            return
        collector.abort()
        target = self._targets[index]
        self._emit(target, phase="aborted")

    def retry_targets(self, indices: list[int]) -> None:
        """Drop batches for the given targets so :meth:`begin_target` starts fresh."""
        for idx in indices:
            self._require_target(idx)
            self._collectors.pop(idx, None)

    def finalize(self) -> CalibrationOutcome:
        """Fit the calibration model, evaluate quality, package the profile."""
        if not self.is_collection_complete:
            raise CalibrationError(
                "finalize called before every target completed collection"
            )
        self._phase = SessionPhase.FITTING
        batches = [self._collectors[t.index].batch() for t in self._targets]
        samples, targets, indices = flatten_batches(batches)

        train_idx, holdout_idx = hold_out_split(
            samples,
            targets,
            indices,
            hold_out_fraction=self.hold_out_fraction,
            seed=self.hold_out_seed,
        )
        model = self._build_model()
        train_samples = [samples[i] for i in train_idx]
        train_targets = [targets[i] for i in train_idx]
        try:
            model.fit(train_samples, train_targets)
        except CalibrationError:
            self._phase = SessionPhase.FAILED
            raise

        holdout_samples = [samples[i] for i in holdout_idx]
        holdout_targets = [targets[i] for i in holdout_idx]
        holdout_indices = [indices[i] for i in holdout_idx]
        holdout_report = evaluate_calibration(
            model,
            holdout_samples,
            holdout_targets,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            target_indices=holdout_indices,
        )
        full_report = evaluate_calibration(
            model,
            samples,
            targets,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            target_indices=indices,
        )
        accepted = holdout_report.mean_error_px <= self.max_residual_px
        bad = identify_bad_targets(full_report, threshold_px=self.max_residual_px)

        profile = build_profile_from_report(
            profile_id=self.profile_id,
            model=model,
            report=full_report,
            created_at_wall_s=self.clock.wall(),
        )
        for target in self._targets:
            self._emit(target, phase="done")
        self._phase = SessionPhase.COMPLETE
        return CalibrationOutcome(
            profile=profile,
            report=full_report,
            holdout_report=holdout_report,
            accepted=accepted,
            bad_targets=bad,
        )

    # ------------------------------------------------------------------ internals
    def _build_model(self) -> AffineCalibration | PolynomialCalibration | RidgeCalibration:
        return build_calibration_model(
            self.model_kind,
            profile_id=self.profile_id,
            polynomial_degree=self.polynomial_degree,
            ridge_alpha=self.ridge_alpha,
        )

    def _require_target(self, index: int) -> CalibrationTarget:
        if not 0 <= index < len(self._targets):
            raise CalibrationError(
                f"target index {index} out of range [0, {len(self._targets)})"
            )
        return self._targets[index]

    def _emit(self, target: CalibrationTarget, *, phase: CalibrationPhase) -> None:
        if self.publisher is None:
            return
        event = CalibrationProgress(
            index=target.index,
            total=len(self._targets),
            target_x=target.nx,
            target_y=target.ny,
            phase=phase,
        )
        self.publisher(event)
