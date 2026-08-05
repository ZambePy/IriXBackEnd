"""Exponential moving average -- the baseline smoother.

Kept as a reference to compare One Euro against. Never the production
choice: EMA has to trade "responsive during saccades" for "still during
fixation" via a single fixed alpha, and one number cannot serve both
regimes. One Euro solves that by making alpha frequency-adaptive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irisflow.filtering.base import SignalSample

__all__ = ["EmaFilter"]


@dataclass
class EmaFilter:
    """Standard first-order EMA on ``(x, y)``.

    ``alpha=1.0`` disables smoothing (passthrough); ``alpha=0`` freezes
    on the first sample.
    """

    alpha: float = 0.4
    _last: SignalSample | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    def step(self, sample: SignalSample) -> SignalSample:
        last = self._last
        if last is None:
            self._last = sample
            return sample
        x = self.alpha * sample.x + (1.0 - self.alpha) * last.x
        y = self.alpha * sample.y + (1.0 - self.alpha) * last.y
        smoothed = SignalSample(x=x, y=y, timestamp=sample.timestamp)
        self._last = smoothed
        return smoothed

    def reset(self) -> None:
        self._last = None
