"""One Euro filter -- adaptive low-pass over a noisy 2D signal.

Reference: Casiez, Roussel & Vogel, "1e Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.
https://cristal.univ-lille.fr/~casiez/1euro/

Intuition: for slow motion (fixation) use aggressive smoothing
(low cutoff = low pass); for fast motion (saccade) reduce smoothing
(raise cutoff toward the raw signal). The cutoff is modulated by the
speed of the smoothed derivative -- hence "adaptive".

Parameters:

* ``min_cutoff`` (Hz): base cutoff frequency at zero speed. Lower is
  smoother at fixation but adds lag; raise if the user complains of
  drift.
* ``beta``: how strongly the cutoff scales with speed. Higher means
  faster response to saccades; too high and one-euro degenerates to
  passthrough. SPRINTS default 0.007 works for most webcams.
* ``d_cutoff`` (Hz): cutoff for the derivative estimate itself. 1.0 Hz
  is the reference recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from irisflow.filtering.base import SignalSample

__all__ = ["OneEuroFilter"]


def _low_pass_alpha(cutoff: float, dt: float) -> float:
    """Standard first-order low-pass alpha from cutoff frequency and dt."""
    if cutoff <= 0 or dt <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuroFilter:
    """Vector One Euro filter over (x, y).

    Args:
        min_cutoff: Hz; base low-pass cutoff (see module docstring).
        beta: cutoff/speed sensitivity coefficient.
        d_cutoff: Hz; low-pass cutoff for the derivative estimate.
    """

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    _last_ts: float | None = field(default=None, init=False)
    _last_x: float | None = field(default=None, init=False)
    _last_y: float | None = field(default=None, init=False)
    _last_dx: float = field(default=0.0, init=False)
    _last_dy: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0:
            raise ValueError(f"min_cutoff must be > 0, got {self.min_cutoff}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.d_cutoff <= 0:
            raise ValueError(f"d_cutoff must be > 0, got {self.d_cutoff}")

    def step(self, sample: SignalSample) -> SignalSample:
        if self._last_ts is None or self._last_x is None or self._last_y is None:
            self._last_ts = sample.timestamp
            self._last_x = sample.x
            self._last_y = sample.y
            self._last_dx = 0.0
            self._last_dy = 0.0
            return sample

        dt = sample.timestamp - self._last_ts
        if dt <= 0:
            # Non-monotonic timestamps: reuse last state, do not advance.
            return SignalSample(x=self._last_x, y=self._last_y, timestamp=sample.timestamp)

        # 1) Estimate derivative, smoothed.
        raw_dx = (sample.x - self._last_x) / dt
        raw_dy = (sample.y - self._last_y) / dt
        a_d = _low_pass_alpha(self.d_cutoff, dt)
        dx = a_d * raw_dx + (1.0 - a_d) * self._last_dx
        dy = a_d * raw_dy + (1.0 - a_d) * self._last_dy
        speed = math.hypot(dx, dy)

        # 2) Speed-modulated cutoff.
        cutoff = self.min_cutoff + self.beta * speed
        a = _low_pass_alpha(cutoff, dt)
        x = a * sample.x + (1.0 - a) * self._last_x
        y = a * sample.y + (1.0 - a) * self._last_y

        self._last_ts = sample.timestamp
        self._last_x = x
        self._last_y = y
        self._last_dx = dx
        self._last_dy = dy
        return SignalSample(x=x, y=y, timestamp=sample.timestamp)

    def reset(self) -> None:
        self._last_ts = None
        self._last_x = None
        self._last_y = None
        self._last_dx = 0.0
        self._last_dy = 0.0
