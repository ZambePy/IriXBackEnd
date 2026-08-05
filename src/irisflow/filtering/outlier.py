"""Reject physically implausible jumps.

Human saccades top out around 900 deg/s. On a 1080p 24" monitor that
maps to roughly 5000-7000 px/s at typical viewing distance. A gaze
sample that would require the eye to move faster than
``max_velocity_px_per_s`` is almost certainly a detection glitch --
freeze the cursor on the previous position instead of teleporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irisflow.filtering.base import SignalSample

__all__ = ["OutlierRejector"]


@dataclass
class OutlierRejector:
    """Freeze the cursor when the implied velocity is unphysical.

    Args:
        max_velocity_px_per_s: Anything above this instantaneous velocity
            is treated as an outlier. Default 6000 px/s -- matches the
            SPRINTS default and covers real sacadas even on 4K displays.
    """

    max_velocity_px_per_s: float = 6000.0
    _last: SignalSample | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.max_velocity_px_per_s <= 0:
            raise ValueError(
                f"max_velocity_px_per_s must be > 0, got {self.max_velocity_px_per_s}"
            )

    def step(self, sample: SignalSample) -> SignalSample:
        last = self._last
        if last is None:
            self._last = sample
            return sample
        dt = sample.timestamp - last.timestamp
        if dt <= 0:
            # Same-timestamp or backwards time -- ignore the sample.
            return last
        dx = sample.x - last.x
        dy = sample.y - last.y
        velocity = (dx * dx + dy * dy) ** 0.5 / dt
        if velocity > self.max_velocity_px_per_s:
            # Freeze on the last accepted sample.
            return last
        self._last = sample
        return sample

    def reset(self) -> None:
        self._last = None
