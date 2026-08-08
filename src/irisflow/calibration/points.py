"""Calibration target grids.

The screen is sampled at a fixed set of normalized targets ``(nx, ny) ∈
[0, 1]²`` — 5, 9 or 13 points — with a configurable margin so that no
target sits exactly on an unreachable screen edge (the corners of many
webcams distort the ROI enough that the CNN degrades).

Pure functions, no I/O. The rendering choice — real screen, WebSocket
message, printed matplotlib figure — lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CalibrationTarget",
    "TargetCount",
    "generate_targets",
]


TargetCount = Literal[5, 9, 13]


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """Normalized ``(nx, ny)`` position of one calibration dot.

    ``index`` uniquely identifies the target within a session so a
    collector can associate every sample back to the right dot without
    keeping a parallel list.
    """

    index: int
    nx: float
    ny: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.nx <= 1.0 or not 0.0 <= self.ny <= 1.0:
            raise ValueError(f"target coordinates must be in [0,1], got ({self.nx}, {self.ny})")
        if self.index < 0:
            raise ValueError(f"target index must be >= 0, got {self.index}")

    def as_tuple(self) -> tuple[float, float]:
        """Return ``(nx, ny)`` as a plain tuple — what the fit routines want."""
        return (self.nx, self.ny)


def generate_targets(count: TargetCount, *, margin: float = 0.08) -> list[CalibrationTarget]:
    """Return the canonical grid for ``count`` targets.

    Args:
        count: One of ``5``, ``9`` or ``13``.
        margin: Fraction of the screen kept clear on every edge. ``0.08``
            keeps targets out of the last 8% strip on each side — a safe
            default given typical webcam FoV and eye-tracking accuracy near
            screen borders.

    Layouts:

    * ``5``: four corners + center - the classic quick calibration.
    * ``9``: 3 by 3 grid - the SPRINTS default, best precision/time ratio.
    * ``13``: 3 by 3 grid plus four extra midpoints along the top and bottom
      edges - for fine-tuning fovea offsets that vary the most vertically.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin}")

    lo = margin
    hi = 1.0 - margin
    mid = 0.5

    if count == 5:
        raw: list[tuple[float, float]] = [
            (lo, lo),
            (hi, lo),
            (mid, mid),
            (lo, hi),
            (hi, hi),
        ]
    elif count == 9:
        raw = [
            (lo, lo),
            (mid, lo),
            (hi, lo),
            (lo, mid),
            (mid, mid),
            (hi, mid),
            (lo, hi),
            (mid, hi),
            (hi, hi),
        ]
    elif count == 13:
        raw = [
            (lo, lo),
            (mid, lo),
            (hi, lo),
            (lo, mid),
            (mid, mid),
            (hi, mid),
            (lo, hi),
            (mid, hi),
            (hi, hi),
            ((lo + mid) / 2, lo),
            ((mid + hi) / 2, lo),
            ((lo + mid) / 2, hi),
            ((mid + hi) / 2, hi),
        ]
    else:  # pragma: no cover — Literal already restricts this at type-check time
        raise ValueError(f"unsupported target count {count!r}, expected 5|9|13")

    return [CalibrationTarget(i, nx, ny) for i, (nx, ny) in enumerate(raw)]
