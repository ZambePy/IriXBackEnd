"""Sample collector for one calibration target.

Feeds :class:`~irisflow.core.types.RawGaze` frames one by one; discards
samples until (a) the user has had time to fixate on the target
(``stabilization_ms``) and (b) the incoming stream is stable enough to
believe the samples are on-target (mean-absolute deviation below
``stability_threshold``). Then collects ``samples_per_point`` raw
samples, ignoring frames flagged as face-lost by the caller.

Pure domain code: no bus, no clock — the clock is passed in via
timestamps on each :meth:`push_sample` call, exactly like
:class:`~irisflow.core.interfaces.Filter` receives ``timestamp`` from
the pipeline runner. That's what makes it testable with
:class:`~irisflow.core.clock.FakeClock` upstream.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from irisflow.core.exceptions import CalibrationError
from irisflow.core.types import RawGaze

__all__ = [
    "CollectorState",
    "TargetCollector",
    "TargetSampleBatch",
]


class CollectorState(StrEnum):
    """Where the collector is in its per-target lifecycle."""

    WAITING = "waiting"  # target just shown, waiting for stabilization
    COLLECTING = "collecting"  # actively taking samples
    DONE = "done"  # enough samples collected
    ABORTED = "aborted"  # caller decided to abandon this target


@dataclass(frozen=True, slots=True)
class TargetSampleBatch:
    """All good samples for one calibration target."""

    target_index: int
    target_nx: float
    target_ny: float
    samples: list[RawGaze] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.samples)


@dataclass
class TargetCollector:
    """State machine for one target's sample collection.

    Args:
        target_index: Index of the target within the session (used to
            attribute samples back on save/eval).
        target_nx, target_ny: Normalized position of the target.
        samples_per_point: How many good samples to keep before declaring
            :attr:`CollectorState.DONE`.
        stabilization_ms: Milliseconds to wait after the target is shown
            before the collector starts accepting samples.
        stability_window: How many recent samples to look at when
            deciding "is the user fixated yet?".
        stability_threshold: Max mean absolute deviation (normalized
            coords) allowed inside the stability window before samples
            are accepted. ``0.05`` = 5% of the frame — generous enough
            for a webcam at 30 FPS but strict enough to reject sacadas.
    """

    target_index: int
    target_nx: float
    target_ny: float
    samples_per_point: int = 30
    stabilization_ms: int = 600
    stability_window: int = 5
    stability_threshold: float = 0.05

    _state: CollectorState = field(default=CollectorState.WAITING, init=False)
    _shown_at_ms: float | None = field(default=None, init=False)
    _accepted: list[RawGaze] = field(default_factory=list, init=False)
    _recent: deque[tuple[float, float]] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        if self.samples_per_point < 1:
            raise CalibrationError(f"samples_per_point must be >= 1, got {self.samples_per_point}")
        if self.stabilization_ms < 0:
            raise CalibrationError(f"stabilization_ms must be >= 0, got {self.stabilization_ms}")
        if self.stability_window < 1:
            raise CalibrationError(f"stability_window must be >= 1, got {self.stability_window}")
        if self.stability_threshold <= 0:
            raise CalibrationError(
                f"stability_threshold must be > 0, got {self.stability_threshold}"
            )
        self._recent = deque(maxlen=self.stability_window)

    # ------------------------------------------------------------------ API
    @property
    def state(self) -> CollectorState:
        return self._state

    @property
    def collected(self) -> int:
        return len(self._accepted)

    @property
    def is_done(self) -> bool:
        return self._state == CollectorState.DONE

    def force_done(self) -> None:
        """Short-circuit collection: accept whatever samples are present as final.

        Used when the caller (usually a CLI driver on a wall-clock budget)
        wants to move on before ``samples_per_point`` is reached. At least
        one sample must have been accepted; otherwise use :meth:`abort`.
        """
        if not self._accepted:
            raise CalibrationError(
                "force_done requires at least one accepted sample; use abort() instead"
            )
        self._state = CollectorState.DONE

    def mark_shown(self, timestamp_s: float) -> None:
        """Record the moment the target became visible to the user."""
        if self._state != CollectorState.WAITING:
            raise CalibrationError(
                f"mark_shown called in state {self._state.value}, expected 'waiting'"
            )
        self._shown_at_ms = float(timestamp_s) * 1000.0

    def abort(self) -> None:
        """Give up on this target — session decides whether to retry it later."""
        self._state = CollectorState.ABORTED

    def push_sample(self, sample: RawGaze | None, timestamp_s: float) -> None:
        """Feed one frame's :class:`RawGaze` (or ``None`` for face-lost).

        Face-lost frames don't crash the collector; they simply clear
        the recent-sample window (the user might be repositioning) and
        force stability to be re-established before more samples are
        accepted. This is what keeps a single blink from poisoning the
        calibration data.
        """
        if self._state in (CollectorState.DONE, CollectorState.ABORTED):
            return
        if self._shown_at_ms is None:
            raise CalibrationError(
                "push_sample called before mark_shown — cannot measure stabilization"
            )

        if sample is None:
            self._recent.clear()
            return

        elapsed_ms = float(timestamp_s) * 1000.0 - self._shown_at_ms
        if elapsed_ms < self.stabilization_ms:
            # Still fixating; just track for stability decisions.
            self._recent.append((sample.x, sample.y))
            return

        self._recent.append((sample.x, sample.y))
        if len(self._recent) < self.stability_window:
            return

        if not self._is_stable():
            return

        self._state = CollectorState.COLLECTING
        self._accepted.append(sample)
        if len(self._accepted) >= self.samples_per_point:
            self._state = CollectorState.DONE

    def batch(self) -> TargetSampleBatch:
        """Return the accepted samples as a batch — safe to call at any state."""
        return TargetSampleBatch(
            target_index=self.target_index,
            target_nx=self.target_nx,
            target_ny=self.target_ny,
            samples=list(self._accepted),
        )

    # ------------------------------------------------------------------ internals
    def _is_stable(self) -> bool:
        """Return true if the sample deque is tight enough to trust."""
        xs = [x for x, _ in self._recent]
        ys = [y for _, y in self._recent]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        mad = sum(abs(x - mean_x) + abs(y - mean_y) for x, y in self._recent) / len(self._recent)
        return mad <= self.stability_threshold


def flatten_batches(
    batches: Sequence[TargetSampleBatch],
) -> tuple[list[RawGaze], list[tuple[float, float]], list[int]]:
    """Unpack a per-target list of batches into three aligned arrays.

    Returns ``(samples, targets, target_indices)`` in the shape every
    :class:`~irisflow.core.interfaces.CalibrationModel.fit` and
    :func:`~irisflow.calibration.quality.evaluate_calibration` expect.
    """
    samples: list[RawGaze] = []
    targets: list[tuple[float, float]] = []
    indices: list[int] = []
    for batch in batches:
        for s in batch.samples:
            samples.append(s)
            targets.append((batch.target_nx, batch.target_ny))
            indices.append(batch.target_index)
    return samples, targets, indices
