"""Safety policy for cursor control (Sprint 10).

The safety layer sits between the pipeline and the OS. It decides:

* **Can the cursor move right now?** — no when the pipeline is in
  ``PAUSED`` or ``LOST``, when the safety layer has been paused (kill
  switch, watchdog, face-lost auto-pause) or when the pointer sits in
  the user-configured rest zone.
* **Can a dwell click fire?** — same rules as movement, plus the
  cursor must currently be enabled by config.

Concrete threading (kill-switch hotkey listener, watchdog thread) lives
in dedicated helpers. Everything here is deterministic and testable
with :class:`~irisflow.core.clock.FakeClock`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from irisflow.core.clock import Clock, SystemClock
from irisflow.core.events import PipelineState

__all__ = [
    "PauseReason",
    "RestZone",
    "SafetyGate",
    "SafetySnapshot",
    "Watchdog",
]


PauseReason = Literal["kill_switch", "face_lost", "watchdog", "manual"]


@dataclass(frozen=True, slots=True)
class RestZone:
    """Rectangular area where clicks are suppressed.

    A rest zone of ``(0, 0, 0, 0)`` (or any zero-area rectangle) is a
    convention for "no rest zone" — matches the YAML default.
    """

    x: int
    y: int
    w: int
    h: int

    def is_enabled(self) -> bool:
        return self.w > 0 and self.h > 0

    def contains(self, px: int, py: int) -> bool:
        if not self.is_enabled():
            return False
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


@dataclass(frozen=True, slots=True)
class SafetySnapshot:
    """Debug snapshot of the safety gate state."""

    pipeline_state: PipelineState
    paused: bool
    pause_reason: PauseReason | None
    face_lost_since_s: float | None


class SafetyGate:
    """Decide whether the cursor may move / click on this tick.

    Args:
        pause_on_face_lost_ms: After the face has been absent for this
            long, the gate auto-pauses with reason ``"face_lost"``. Zero
            disables the auto-pause (cursor freezes on LOST anyway).
        rest_zone: Screen rectangle where clicks are suppressed but
            movement is still allowed. ``(0,0,0,0)`` disables the zone.
        clock: Time source. Real clock in production; :class:`FakeClock`
            in tests.
    """

    __slots__ = (
        "_clock",
        "_face_lost_since_s",
        "_lock",
        "_pause_on_face_lost_ms",
        "_pause_reason",
        "_paused",
        "_pipeline_state",
        "_rest_zone",
    )

    def __init__(
        self,
        *,
        pause_on_face_lost_ms: int,
        rest_zone: RestZone,
        clock: Clock | None = None,
    ) -> None:
        if pause_on_face_lost_ms < 0:
            raise ValueError(
                f"pause_on_face_lost_ms must be >= 0, got {pause_on_face_lost_ms}"
            )
        self._pause_on_face_lost_ms = pause_on_face_lost_ms
        self._rest_zone = rest_zone
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.Lock()
        self._pipeline_state: PipelineState = PipelineState.IDLE
        self._paused = False
        self._pause_reason: PauseReason | None = None
        self._face_lost_since_s: float | None = None

    # ------------------------------------------------------------------ observers
    def on_pipeline_state(self, state: PipelineState) -> None:
        """Track the pipeline state so :meth:`can_move` knows LOST/PAUSED."""
        with self._lock:
            self._pipeline_state = state
            if state != PipelineState.LOST:
                self._face_lost_since_s = None
            if state != PipelineState.PAUSED and self._pause_reason == "manual":
                # If the pipeline itself resumed, clear the manual pause flag.
                self._paused = False
                self._pause_reason = None

    def on_face_lost(self) -> None:
        """Start (or continue) tracking how long the face has been missing."""
        with self._lock:
            if self._face_lost_since_s is None:
                self._face_lost_since_s = self._clock.monotonic()
            self._maybe_face_lost_pause_locked()

    def on_face_acquired(self) -> None:
        """Clear the face-lost timer and auto-pause if it was set for that."""
        with self._lock:
            self._face_lost_since_s = None
            if self._paused and self._pause_reason == "face_lost":
                self._paused = False
                self._pause_reason = None

    # ------------------------------------------------------------------ commands
    def trigger_pause(self, reason: PauseReason) -> bool:
        """Explicitly pause the cursor. Returns True if state changed."""
        with self._lock:
            if self._paused:
                return False
            self._paused = True
            self._pause_reason = reason
            return True

    def resume(self) -> bool:
        """Release any active pause. Returns True if state changed."""
        with self._lock:
            if not self._paused:
                return False
            self._paused = False
            self._pause_reason = None
            return True

    def tick(self) -> None:
        """Refresh time-based pause conditions (called each pipeline tick)."""
        with self._lock:
            self._maybe_face_lost_pause_locked()

    # ------------------------------------------------------------------ queries
    def can_move(self) -> bool:
        with self._lock:
            return not self._paused and self._pipeline_state == PipelineState.TRACKING

    def can_click(self, px: int, py: int) -> bool:
        with self._lock:
            if self._paused or self._pipeline_state != PipelineState.TRACKING:
                return False
            return not self._rest_zone.contains(px, py)

    def snapshot(self) -> SafetySnapshot:
        with self._lock:
            return SafetySnapshot(
                pipeline_state=self._pipeline_state,
                paused=self._paused,
                pause_reason=self._pause_reason,
                face_lost_since_s=self._face_lost_since_s,
            )

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def pause_reason(self) -> PauseReason | None:
        with self._lock:
            return self._pause_reason

    # ------------------------------------------------------------------ internals
    def _maybe_face_lost_pause_locked(self) -> None:
        """Assumes ``self._lock`` is held. Auto-pause when face-lost is old."""
        if self._pause_on_face_lost_ms == 0:
            return
        if self._face_lost_since_s is None:
            return
        elapsed_ms = (self._clock.monotonic() - self._face_lost_since_s) * 1000.0
        if elapsed_ms >= self._pause_on_face_lost_ms and not self._paused:
            self._paused = True
            self._pause_reason = "face_lost"


class Watchdog:
    """Fire a callback when no ``kick()`` has landed within ``timeout_ms``.

    Purpose (SPRINTS §10): if the pipeline stalls — a deadlock, a wedged
    detector, a bad GPU state — the OS cursor must not stay under gaze
    control. The watchdog runs on its own daemon thread; when it fires
    the caller's ``on_stall`` typically calls
    :meth:`SafetyGate.trigger_pause("watchdog")` and disables the cursor
    controller.
    """

    __slots__ = (
        "_clock",
        "_last_kick_s",
        "_lock",
        "_on_stall",
        "_stopped",
        "_thread",
        "_timeout_ms",
    )

    def __init__(
        self,
        *,
        timeout_ms: int,
        on_stall: Callable[[], None],
        clock: Clock | None = None,
    ) -> None:
        if timeout_ms < 0:
            raise ValueError(f"timeout_ms must be >= 0, got {timeout_ms}")
        self._timeout_ms = timeout_ms
        self._on_stall = on_stall
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.Lock()
        self._last_kick_s = self._clock.monotonic()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def kick(self) -> None:
        """Mark the pipeline as alive at the current instant."""
        with self._lock:
            self._last_kick_s = self._clock.monotonic()

    def check(self) -> bool:
        """Return True and fire the callback if the pipeline has stalled."""
        if self._timeout_ms == 0:
            return False
        with self._lock:
            elapsed_ms = (self._clock.monotonic() - self._last_kick_s) * 1000.0
        if elapsed_ms >= self._timeout_ms:
            self._on_stall()
            return True
        return False

    def start(self, poll_interval_s: float = 0.1) -> None:
        """Start a background thread that calls :meth:`check` periodically.

        For unit tests, call :meth:`check` directly with a :class:`FakeClock`
        instead of relying on the thread.
        """
        if self._timeout_ms == 0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopped.clear()

        def _loop() -> None:
            while not self._stopped.wait(poll_interval_s):
                try:
                    self.check()
                except Exception:  # pragma: no cover - defensive
                    return

        self._thread = threading.Thread(
            target=_loop, name="irisflow-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None
