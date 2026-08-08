"""Glue between the event bus and the cursor / dwell / safety layer.

Lives in :mod:`irisflow.pipeline` (not in :mod:`irisflow.control`) because
it wires two neighbours together: the ``pipeline`` layer already knows
about the bus, whereas the ``control`` layer sits below ``pipeline`` and
must not import it (import-linter contract).

Responsibilities:

* subscribe to :class:`~irisflow.core.events.GazeUpdated`,
  :class:`~irisflow.core.events.StateChanged`,
  :class:`~irisflow.core.events.FaceLost` and
  :class:`~irisflow.core.events.FaceAcquired`;
* consult the :class:`~irisflow.control.safety.SafetyGate` on every
  gaze tick to decide whether to move the cursor and whether to feed
  the :class:`~irisflow.control.dwell.DwellClicker`;
* publish :class:`~irisflow.core.events.DwellProgress` /
  :class:`~irisflow.core.events.DwellClick` /
  :class:`~irisflow.core.events.SafetyPaused` /
  :class:`~irisflow.core.events.SafetyResumed` back on the bus so that
  future frontends and telemetry sinks see the same signal;
* enable / disable the underlying :class:`CursorController` at the right
  moments so that a stalled pipeline or a kill-switch press releases
  the mouse in under 100 ms (SPRINTS §10 DoD).
"""

from __future__ import annotations

from irisflow.control.dwell import DwellClicker
from irisflow.control.kill_switch import KillSwitchListener
from irisflow.control.safety import PauseReason, SafetyGate, Watchdog
from irisflow.core.clock import Clock, SystemClock
from irisflow.core.events import (
    DwellClick,
    DwellProgress,
    Event,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    SafetyPaused,
    SafetyResumed,
    StateChanged,
)
from irisflow.core.interfaces import CursorController
from irisflow.logging import get_logger
from irisflow.pipeline.bus import EventBus

__all__ = ["ControlSink"]


class ControlSink:
    """Bus consumer that drives the cursor + dwell click + safety layer.

    Args:
        bus: The pipeline event bus.
        cursor: The controller that actually moves the OS cursor (or a
            :class:`~irisflow.control.noop.NoOpCursor` in headless mode).
        safety: Pre-built safety gate.
        dwell: Pre-built dwell clicker.
        kill_switch: Optional kill-switch listener; if provided it is
            started by :meth:`start` and stopped by :meth:`stop`.
        watchdog: Optional :class:`Watchdog`; when provided it is started
            by :meth:`start` and stopped by :meth:`stop`.
        cursor_enabled_on_start: Whether the cursor is enabled the
            moment :meth:`start` runs. ``False`` by default so that
            ``irisflow run`` (no ``--cursor``) never nudges the mouse
            accidentally.
        clock: Time source for the ``GazeUpdated`` timestamp math when
            dwell needs to know "now".
    """

    __slots__ = (
        "_bus",
        "_clock",
        "_cursor",
        "_cursor_enabled_on_start",
        "_dwell",
        "_kill_switch",
        "_log",
        "_safety",
        "_started",
        "_watchdog",
    )

    def __init__(
        self,
        *,
        bus: EventBus,
        cursor: CursorController,
        safety: SafetyGate,
        dwell: DwellClicker,
        kill_switch: KillSwitchListener | None = None,
        watchdog: Watchdog | None = None,
        cursor_enabled_on_start: bool = False,
        clock: Clock | None = None,
    ) -> None:
        self._bus = bus
        self._cursor = cursor
        self._safety = safety
        self._dwell = dwell
        self._kill_switch = kill_switch
        self._watchdog = watchdog
        self._cursor_enabled_on_start = cursor_enabled_on_start
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._log = get_logger("irisflow.pipeline.control_sink")
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Subscribe to bus events and start any background threads."""
        if self._started:
            return
        self._bus.subscribe(GazeUpdated, self._on_gaze)
        self._bus.subscribe(StateChanged, self._on_state)
        self._bus.subscribe(FaceLost, self._on_face_lost)
        self._bus.subscribe(FaceAcquired, self._on_face_acquired)
        if self._cursor_enabled_on_start:
            self._cursor.enable()
        else:
            self._cursor.disable()
        if self._kill_switch is not None:
            self._kill_switch.start()
        if self._watchdog is not None:
            self._watchdog.start()
        self._started = True

    def stop(self) -> None:
        """Unsubscribe, disable the cursor, stop background threads."""
        if not self._started:
            return
        self._bus.unsubscribe(GazeUpdated, self._on_gaze)
        self._bus.unsubscribe(StateChanged, self._on_state)
        self._bus.unsubscribe(FaceLost, self._on_face_lost)
        self._bus.unsubscribe(FaceAcquired, self._on_face_acquired)
        self._cursor.disable()
        if self._kill_switch is not None:
            self._kill_switch.stop()
        if self._watchdog is not None:
            self._watchdog.stop()
        self._started = False

    # ------------------------------------------------------------------ kill switch
    def on_kill_switch(self) -> None:
        """Callback wired to :class:`KillSwitchListener` — pause + disable."""
        self._trigger_pause("kill_switch")

    def on_watchdog_stall(self) -> None:
        """Callback wired to :class:`Watchdog` — pause + disable on stall."""
        self._trigger_pause("watchdog")

    def watchdog_kick(self) -> None:
        """No-op when no watchdog is wired; feeds the timer otherwise."""
        if self._watchdog is not None:
            self._watchdog.kick()

    # ------------------------------------------------------------------ bus handlers
    def _on_state(self, event: Event) -> None:
        if not isinstance(event, StateChanged):
            return
        self._safety.on_pipeline_state(event.current)

    def _on_face_lost(self, event: Event) -> None:
        if not isinstance(event, FaceLost):
            return
        self._safety.on_face_lost()
        if self._safety.is_paused and self._safety.pause_reason == "face_lost":
            self._cursor.disable()
            self._bus.publish(
                SafetyPaused(timestamp=self._clock.monotonic(), reason="face_lost")
            )

    def _on_face_acquired(self, event: Event) -> None:
        if not isinstance(event, FaceAcquired):
            return
        was_paused = self._safety.is_paused and self._safety.pause_reason == "face_lost"
        self._safety.on_face_acquired()
        if was_paused and self._cursor_enabled_on_start:
            self._cursor.enable()
            self._bus.publish(SafetyResumed(timestamp=self._clock.monotonic()))

    def _on_gaze(self, event: Event) -> None:
        if not isinstance(event, GazeUpdated):
            return
        # Tick safety so face-lost timeout can trip mid-stream.
        self._safety.tick()
        if self._safety.is_paused:
            return
        if not self._safety.can_move():
            return
        self._cursor.move(event.px, event.py)
        if not self._cursor.is_enabled:
            return
        decision = self._dwell.on_sample(event.px, event.py, event.timestamp)
        if decision.phase != "refractory":
            self._bus.publish(
                DwellProgress(
                    frame_id=event.frame_id,
                    timestamp=event.timestamp,
                    px=event.px,
                    py=event.py,
                    progress=decision.progress,
                    radius_px=self._dwell.params.radius_px,
                )
            )
        if decision.click_at is not None:
            cx, cy = decision.click_at
            if self._safety.can_click(cx, cy):
                self._cursor.click("left")
                self._bus.publish(
                    DwellClick(
                        frame_id=event.frame_id,
                        timestamp=event.timestamp,
                        px=cx,
                        py=cy,
                        button="left",
                    )
                )
            else:
                # Safety refused the click (rest zone, kill switch, etc.):
                # drop the refractory so the user gets a fresh dwell the
                # next time they move to a legal area.
                self._dwell.reset()

    # ------------------------------------------------------------------ internals
    def _trigger_pause(self, reason: PauseReason) -> None:
        changed = self._safety.trigger_pause(reason)
        self._cursor.disable()
        if changed:
            self._log.warning("safety.paused", reason=reason)
            self._bus.publish(
                SafetyPaused(timestamp=self._clock.monotonic(), reason=reason)
            )
