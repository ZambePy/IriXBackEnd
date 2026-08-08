"""Fixation vs saccade classification.

Human oculomotor behaviour comes in two clean regimes: fixation (eye
holds a point, ~30-300 ms) and saccade (ballistic jump, ~20-100 ms).
Distinguishing them lets the Sprint 10 dwell logic count "how long
have I been looking at this button?" without a raw stability check
(which would be fooled by a slow drift back to the same point).

Threshold-only classifier: velocity below the threshold => fixation.
More sophisticated methods (I-VT with dispersion, HMM) exist but add
enough complexity to not be worth it at 30 Hz webcam sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from irisflow.filtering.base import SignalSample

__all__ = ["FixationClassifier", "FixationDecision"]


@dataclass(frozen=True, slots=True)
class FixationDecision:
    """Result of one classification step."""

    velocity_px_per_s: float
    is_fixation: bool


@dataclass
class FixationClassifier:
    """Label each smoothed sample as fixation or saccade.

    Args:
        velocity_threshold_px_per_s: Below this speed the sample is
            classified as fixation. 40 px/s is a good default for typical
            webcam+monitor setups; sharper monitors may want lower.
    """

    velocity_threshold_px_per_s: float = 40.0
    _last: SignalSample | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.velocity_threshold_px_per_s <= 0:
            raise ValueError(
                f"velocity_threshold_px_per_s must be > 0, got {self.velocity_threshold_px_per_s}"
            )

    def classify(self, sample: SignalSample) -> FixationDecision:
        last = self._last
        self._last = sample
        if last is None:
            # First sample -- no velocity available; conservatively
            # report the point as fixation with zero velocity.
            return FixationDecision(velocity_px_per_s=0.0, is_fixation=True)
        dt = sample.timestamp - last.timestamp
        if dt <= 0:
            return FixationDecision(velocity_px_per_s=0.0, is_fixation=True)
        velocity = math.hypot(sample.x - last.x, sample.y - last.y) / dt
        return FixationDecision(
            velocity_px_per_s=velocity,
            is_fixation=velocity < self.velocity_threshold_px_per_s,
        )

    def reset(self) -> None:
        self._last = None
