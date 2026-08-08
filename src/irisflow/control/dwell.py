"""Dwell-click state machine (Sprint 10).

Pure logic — no threads, no OS calls. The pipeline feeds it one
``(px, py, timestamp)`` per delivered gaze frame; the clicker returns a
:class:`DwellDecision` describing the current progress and whether a
click should fire this tick.

Dwell definition used here (SPRINTS §10 dwell):

* A **cluster** is the anchor pixel where the pointer first entered a
  radius-bounded zone. As long as the current pointer stays within
  ``radius_px`` of the anchor, we accumulate time.
* When accumulated time reaches ``duration_ms`` a click fires at the
  cluster centroid (running mean of the accepted points), not at the
  most recent point — using the last frame is jitter-sensitive.
* After a click the clicker enters a **refractory** window
  (``refractory_ms``) during which no new dwell can start; the pointer
  must first leave the radius. This is the safety net that stops the
  user from double-clicking accidentally when they linger.

The classifier is only meaningful *while the pipeline is TRACKING* — the
:class:`~irisflow.control.safety.SafetyGate` gates every call to
:meth:`on_sample` behind that check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = ["DwellClicker", "DwellDecision", "DwellParams"]


@dataclass(frozen=True, slots=True)
class DwellParams:
    """Configuration bundle — mirrors :class:`~irisflow.config.schema.DwellConfig`.

    Kept as a plain dataclass so this module does not depend on
    :mod:`irisflow.config` (sibling layer).
    """

    radius_px: int = 40
    duration_ms: int = 800
    refractory_ms: int = 400

    def __post_init__(self) -> None:
        if self.radius_px <= 0:
            raise ValueError(f"radius_px must be > 0, got {self.radius_px}")
        if self.duration_ms <= 0:
            raise ValueError(f"duration_ms must be > 0, got {self.duration_ms}")
        if self.refractory_ms < 0:
            raise ValueError(f"refractory_ms must be >= 0, got {self.refractory_ms}")


@dataclass(frozen=True, slots=True)
class DwellDecision:
    """Outcome of one :meth:`DwellClicker.on_sample` call.

    ``progress`` is always in ``[0, 1]``. ``click_at`` is ``None`` unless
    this tick crosses the duration threshold — then it holds the pixel
    at which the click should fire (cluster centroid).
    """

    progress: float
    click_at: tuple[int, int] | None
    phase: Literal["idle", "collecting", "refractory"]


class DwellClicker:
    """Dwell-click accumulator.

    Args:
        params: Radius / duration / refractory (all fixed for the session).
    """

    __slots__ = (
        "_anchor_x",
        "_anchor_y",
        "_dwell_start",
        "_last_click_at",
        "_params",
        "_sum_count",
        "_sum_x",
        "_sum_y",
    )

    def __init__(self, params: DwellParams) -> None:
        self._params = params
        self._reset_cluster()
        self._last_click_at: float | None = None

    # ------------------------------------------------------------------ API
    @property
    def params(self) -> DwellParams:
        return self._params

    @property
    def is_refractory(self) -> bool:
        return self._last_click_at is not None

    def reset(self) -> None:
        """Drop all state — dwell restarts fresh, refractory cleared."""
        self._reset_cluster()
        self._last_click_at = None

    def on_sample(self, px: int, py: int, timestamp_s: float) -> DwellDecision:
        """Feed one gaze sample; returns the current dwell status.

        Args:
            px, py: Screen-space pixel of this frame.
            timestamp_s: Monotonic seconds — same clock the caller uses.
        """
        # Refractory window: clear when the pointer has moved out of the
        # last click's radius. Otherwise stay refractory.
        if self._last_click_at is not None:
            if (
                self._anchor_x is None
                or self._anchor_y is None
                or self._is_outside(px, py, self._anchor_x, self._anchor_y)
            ):
                # Enter refractory-outside: reset cluster but keep the
                # "clicked" marker until the ``refractory_ms`` timer expires.
                self._reset_cluster()
                if (timestamp_s - self._last_click_at) * 1000.0 >= self._params.refractory_ms:
                    self._last_click_at = None
                return DwellDecision(progress=0.0, click_at=None, phase="refractory")
            # Still inside the just-clicked cluster; suppress and keep waiting.
            if (timestamp_s - self._last_click_at) * 1000.0 < self._params.refractory_ms:
                return DwellDecision(progress=0.0, click_at=None, phase="refractory")
            # Refractory expired but the pointer hasn't moved — release the
            # click gate and start a fresh cluster from here.
            self._last_click_at = None
            self._reset_cluster()

        # Not refractory: either extend the existing cluster or open a new one.
        if self._anchor_x is None or self._anchor_y is None:
            self._start_cluster(px, py, timestamp_s)
            return DwellDecision(progress=0.0, click_at=None, phase="collecting")

        if self._is_outside(px, py, self._anchor_x, self._anchor_y):
            # Pointer wandered off — start a new cluster at the new location.
            self._start_cluster(px, py, timestamp_s)
            return DwellDecision(progress=0.0, click_at=None, phase="collecting")

        # Inside the radius: accumulate. Guard against non-monotonic timestamps
        # (never should happen with monotonic clocks, but be defensive).
        assert self._dwell_start is not None
        elapsed_ms = max(0.0, (timestamp_s - self._dwell_start) * 1000.0)
        self._sum_x += px
        self._sum_y += py
        self._sum_count += 1
        progress = min(1.0, elapsed_ms / self._params.duration_ms)
        if elapsed_ms >= self._params.duration_ms:
            centroid = (
                round(self._sum_x / self._sum_count),
                round(self._sum_y / self._sum_count),
            )
            self._last_click_at = timestamp_s
            self._reset_cluster()
            return DwellDecision(progress=1.0, click_at=centroid, phase="collecting")
        return DwellDecision(progress=progress, click_at=None, phase="collecting")

    # ------------------------------------------------------------------ internals
    def _reset_cluster(self) -> None:
        self._anchor_x: int | None = None
        self._anchor_y: int | None = None
        self._dwell_start: float | None = None
        self._sum_x = 0
        self._sum_y = 0
        self._sum_count = 0

    def _start_cluster(self, px: int, py: int, timestamp_s: float) -> None:
        self._anchor_x = px
        self._anchor_y = py
        self._dwell_start = timestamp_s
        self._sum_x = px
        self._sum_y = py
        self._sum_count = 1

    def _is_outside(self, px: int, py: int, anchor_x: int, anchor_y: int) -> bool:
        dx = px - anchor_x
        dy = py - anchor_y
        return math.hypot(dx, dy) > self._params.radius_px
