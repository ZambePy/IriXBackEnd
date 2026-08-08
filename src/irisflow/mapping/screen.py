"""Normalized calibrated gaze -> integer screen coordinate.

The mapper turns ``(nx, ny) in [0, 1]^2`` into a pixel on a specific
:class:`~irisflow.mapping.screen_info.ScreenInfo`, clamping to the
inside of the screen minus a configurable border margin. The margin
matters for ALS: if a target lives literally on the last row of pixels
the cursor can jitter into "off-screen" territory where dwell targets
disappear.

Pure domain code -- no OS access, no bus, testable with plain calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from irisflow.core.geometry import clamp
from irisflow.core.types import CalibratedGaze, ScreenPoint
from irisflow.mapping.screen_info import ScreenInfo

__all__ = ["ScreenMapper"]


@dataclass(frozen=True, slots=True)
class ScreenMapper:
    """Denormalize gaze to screen pixel with a border margin clamp.

    Args:
        screen: The target monitor.
        clamp_margin_px: Pixels of border to keep the cursor away from.
            The mapper never returns coordinates in the margin strip,
            even for gaze that lands outside ``[0, 1]``.
    """

    screen: ScreenInfo
    clamp_margin_px: int = 0

    def __post_init__(self) -> None:
        if self.clamp_margin_px < 0:
            raise ValueError(f"clamp_margin_px must be >= 0, got {self.clamp_margin_px}")
        max_margin_w = self.screen.width_px // 2 - 1
        max_margin_h = self.screen.height_px // 2 - 1
        if self.clamp_margin_px > min(max_margin_w, max_margin_h):
            raise ValueError(
                f"clamp_margin_px {self.clamp_margin_px} leaves no usable area "
                f"on screen {self.screen.width_px}x{self.screen.height_px}"
            )

    def to_screen(self, gaze: CalibratedGaze) -> ScreenPoint:
        """Map ``gaze`` to a pixel inside the safe drawable rectangle."""
        px = self.screen.origin_x_px + round(gaze.x * self.screen.width_px)
        py = self.screen.origin_y_px + round(gaze.y * self.screen.height_px)
        return ScreenPoint(
            px=int(clamp(px, self._min_x, self._max_x)),
            py=int(clamp(py, self._min_y, self._max_y)),
            screen_id=self.screen.screen_id,
        )

    # ------------------------------------------------------------------ props
    @property
    def _min_x(self) -> int:
        return self.screen.origin_x_px + self.clamp_margin_px

    @property
    def _max_x(self) -> int:
        return self.screen.x2 - 1 - self.clamp_margin_px

    @property
    def _min_y(self) -> int:
        return self.screen.origin_y_px + self.clamp_margin_px

    @property
    def _max_y(self) -> int:
        return self.screen.y2 - 1 - self.clamp_margin_px
