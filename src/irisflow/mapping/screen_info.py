"""Screen geometry: resolution, DPI, multi-monitor.

Pure data: this module never queries the real display. In production
the CLI/pipeline layer resolves the actual screen dims (via
``tkinter.Tk().winfo_screenwidth()`` or the config) and hands them to
:class:`ScreenInfo`; in tests we construct fake screens directly.

Kept as a plain dataclass instead of a Pydantic model because the
config layer is where validation belongs -- this file lives inside
:mod:`irisflow.mapping`, a sibling of :mod:`irisflow.config`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MultiScreenLayout", "ScreenInfo"]


@dataclass(frozen=True, slots=True)
class ScreenInfo:
    """One physical display.

    ``screen_id`` uniquely identifies the monitor within a session so
    :class:`~irisflow.core.types.ScreenPoint` carries a reference back to
    the layout that produced it.

    ``dpi`` is optional: only the multi-monitor code needs it, and
    single-monitor setups rarely publish an accurate value. When known
    it lets us convert between pixel error and physical distance --
    handy for reporting fatigue-relevant metrics in the future.
    """

    width_px: int
    height_px: int
    screen_id: int = 0
    dpi: float | None = None
    origin_x_px: int = 0
    origin_y_px: int = 0

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError(
                f"ScreenInfo requires positive dims, got "
                f"{self.width_px}x{self.height_px}"
            )
        if self.dpi is not None and self.dpi <= 0:
            raise ValueError(f"ScreenInfo.dpi must be positive, got {self.dpi}")

    @property
    def x2(self) -> int:
        return self.origin_x_px + self.width_px

    @property
    def y2(self) -> int:
        return self.origin_y_px + self.height_px

    def contains(self, px: int, py: int) -> bool:
        """Return True when ``(px, py)`` lies inside this screen's rectangle."""
        return (
            self.origin_x_px <= px < self.x2
            and self.origin_y_px <= py < self.y2
        )


@dataclass(frozen=True, slots=True)
class MultiScreenLayout:
    """A collection of screens indexed by :attr:`ScreenInfo.screen_id`.

    The pipeline chooses which screen to map gaze to via
    :class:`~irisflow.config.schema.MappingConfig.screen_id`; the layout
    exists so future work can support "gaze went off screen 0, is it on
    screen 1?" -- not needed in Sprint 9 but keeping the seam ready
    avoids a bigger refactor later.
    """

    screens: tuple[ScreenInfo, ...]

    def __post_init__(self) -> None:
        if not self.screens:
            raise ValueError("MultiScreenLayout requires at least one screen")
        seen_ids: set[int] = set()
        for s in self.screens:
            if s.screen_id in seen_ids:
                raise ValueError(
                    f"MultiScreenLayout: duplicate screen_id={s.screen_id}"
                )
            seen_ids.add(s.screen_id)

    def get(self, screen_id: int) -> ScreenInfo:
        for s in self.screens:
            if s.screen_id == screen_id:
                return s
        raise KeyError(f"screen_id {screen_id} not present in layout")

    def primary(self) -> ScreenInfo:
        """Lowest-id screen -- convention for the default target monitor."""
        return min(self.screens, key=lambda s: s.screen_id)
