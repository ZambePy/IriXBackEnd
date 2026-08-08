"""Deterministically re-run the downstream pipeline over a recorded session.

Sprint 11 promise (SPRINTS.md DoD): "replay produces output **identical**
to the original execution (determinism with ``FakeClock``)".

Rather than replay the capture / detection / inference stages — which
depend on hardware and floating-point non-determinism inside third-party
libraries — this module replays only the deterministic tail of the
pipeline: raw gaze → calibration → mapping → filtering → dwell.

Input : a session file that includes at least
        :class:`~irisflow.core.events.RawGazeReady`,
        :class:`~irisflow.core.events.FaceLost`,
        :class:`~irisflow.core.events.FaceAcquired` and
        :class:`~irisflow.core.events.StateChanged` events.
Output: the same sequence of downstream events that the runner would
        have published — :class:`~irisflow.core.events.GazeUpdated`,
        :class:`~irisflow.core.events.DwellProgress`,
        :class:`~irisflow.core.events.DwellClick`,
        :class:`~irisflow.core.events.SafetyPaused`,
        :class:`~irisflow.core.events.SafetyResumed`.

The replayer takes the same collaborator objects the pipeline uses at
runtime (calibration model, mapper, filter chain, cursor, safety gate,
dwell), so tests can assert bit-identical output by wiring the exact
same instances that produced the recording.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from irisflow.control.dwell import DwellClicker
from irisflow.control.safety import SafetyGate
from irisflow.core.clock import FakeClock
from irisflow.core.events import (
    DwellClick,
    DwellProgress,
    Event,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    RawGazeReady,
    SafetyPaused,
    SafetyResumed,
    StateChanged,
)
from irisflow.core.interfaces import CalibrationModel, Filter, ScreenMapper
from irisflow.core.types import CalibratedGaze, RawGaze
from irisflow.logging import get_logger

__all__ = ["ReplayResult", "SessionReplayer"]


@dataclass
class ReplayResult:
    """Everything produced by :meth:`SessionReplayer.replay`."""

    events: list[Event] = field(default_factory=list)
    gaze_updates: list[GazeUpdated] = field(default_factory=list)
    dwell_progress: list[DwellProgress] = field(default_factory=list)
    dwell_clicks: list[DwellClick] = field(default_factory=list)
    safety_paused: list[SafetyPaused] = field(default_factory=list)
    safety_resumed: list[SafetyResumed] = field(default_factory=list)


class SessionReplayer:
    """Re-run the tail of the pipeline over a recorded event stream.

    Args:
        calibration_model: Optional calibration transform. When ``None``
            or unfitted, raw gaze is used verbatim (matches the runner's
            fallback path).
        mapper: Maps calibrated gaze to a screen pixel.
        filter_chain: Applies the same smoothing chain used at runtime.
        safety: Pre-built safety gate — replay must respect the same
            pause/kill/watchdog decisions the runtime made.
        dwell: Pre-built dwell clicker.
        cursor_enabled: Whether dwell clicks fire. Matches the runtime
            ``--cursor`` flag. When ``False`` the replay still emits
            :class:`DwellProgress` inside a non-refractory phase.
    """

    __slots__ = (
        "_calibration_model",
        "_clock",
        "_cursor_enabled",
        "_dwell",
        "_face_present",
        "_filter",
        "_log",
        "_mapper",
        "_safety",
    )

    def __init__(
        self,
        *,
        calibration_model: CalibrationModel | None,
        mapper: ScreenMapper,
        filter_chain: Filter,
        safety: SafetyGate,
        dwell: DwellClicker,
        cursor_enabled: bool = True,
        clock: FakeClock | None = None,
    ) -> None:
        self._calibration_model = calibration_model
        self._mapper = mapper
        self._filter = filter_chain
        self._safety = safety
        self._dwell = dwell
        self._cursor_enabled = cursor_enabled
        self._clock = clock
        self._face_present = True
        self._log = get_logger("irisflow.telemetry.replayer")

    # ------------------------------------------------------------------ API
    def replay(self, events: Iterable[Event]) -> ReplayResult:
        """Feed ``events`` through the pipeline tail and return every
        downstream event the runtime would have emitted, in order.
        """
        result = ReplayResult()
        for event in events:
            self._dispatch(event, result)
        return result

    def replay_seq(self, events: Sequence[Event]) -> ReplayResult:
        return self.replay(events)

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, event: Event, result: ReplayResult) -> None:
        self._sync_clock(event)
        if isinstance(event, StateChanged):
            self._safety.on_pipeline_state(event.current)
            return
        if isinstance(event, FaceLost):
            self._safety.on_face_lost()
            self._safety.tick()
            self._face_present = False
            self._maybe_emit_safety_pause(event.timestamp, result)
            return
        if isinstance(event, FaceAcquired):
            was_paused_face = (
                self._safety.is_paused
                and self._safety.pause_reason == "face_lost"
            )
            self._safety.on_face_acquired()
            self._face_present = True
            if was_paused_face and self._cursor_enabled:
                result.events.append(SafetyResumed(timestamp=event.timestamp))
                result.safety_resumed.append(
                    SafetyResumed(timestamp=event.timestamp)
                )
            return
        if isinstance(event, RawGazeReady):
            self._on_raw_gaze(event, result)
            return
        # Every other event type is metadata / already-recorded — the
        # replayer does not need to reproduce it (report reads the file
        # directly for those).

    # ------------------------------------------------------------------ raw gaze
    def _on_raw_gaze(self, event: RawGazeReady, result: ReplayResult) -> None:
        raw = RawGaze(
            x=event.x,
            y=event.y,
            confidence=event.confidence,
            inference_ms=event.inference_ms,
        )
        if self._calibration_model is not None and self._calibration_model.is_fitted:
            calibrated = self._calibration_model.transform(raw)
        else:
            calibrated = CalibratedGaze(x=raw.x, y=raw.y, profile_id="raw")
        screen_point = self._mapper.to_screen(calibrated)
        smoothed = self._filter.apply(screen_point, event.timestamp)
        gaze = GazeUpdated(
            frame_id=event.frame_id,
            timestamp=event.timestamp,
            px=smoothed.px,
            py=smoothed.py,
            is_fixation=smoothed.is_fixation,
            confidence=raw.confidence,
        )
        result.events.append(gaze)
        result.gaze_updates.append(gaze)

        # ControlSink parity: tick safety, then dwell.
        self._safety.tick()
        if self._safety.is_paused:
            return
        if not self._safety.can_move():
            return
        if not self._cursor_enabled:
            return
        decision = self._dwell.on_sample(gaze.px, gaze.py, gaze.timestamp)
        if decision.phase != "refractory":
            progress = DwellProgress(
                frame_id=event.frame_id,
                timestamp=event.timestamp,
                px=gaze.px,
                py=gaze.py,
                progress=decision.progress,
                radius_px=self._dwell.params.radius_px,
            )
            result.events.append(progress)
            result.dwell_progress.append(progress)
        if decision.click_at is not None:
            cx, cy = decision.click_at
            if self._safety.can_click(cx, cy):
                click = DwellClick(
                    frame_id=event.frame_id,
                    timestamp=event.timestamp,
                    px=cx,
                    py=cy,
                    button="left",
                )
                result.events.append(click)
                result.dwell_clicks.append(click)
            else:
                self._dwell.reset()

    # ------------------------------------------------------------------ clock sync
    def _sync_clock(self, event: Event) -> None:
        """Advance the injected FakeClock to ``event.timestamp`` if any.

        The SafetyGate's face-lost timer and any watchdog rely on the
        clock; without this, replay of a recorded session would never
        see time pass and the timers would never fire.
        """
        if self._clock is None:
            return
        ts = getattr(event, "timestamp", None)
        if not isinstance(ts, (int, float)):
            return
        delta = float(ts) - self._clock.monotonic()
        if delta > 0:
            self._clock.advance(delta)

    # ------------------------------------------------------------------ safety helpers
    def _maybe_emit_safety_pause(self, timestamp: float, result: ReplayResult) -> None:
        if (
            self._safety.is_paused
            and self._safety.pause_reason == "face_lost"
            and self._cursor_enabled
        ):
            paused = SafetyPaused(timestamp=timestamp, reason="face_lost")
            result.events.append(paused)
            result.safety_paused.append(paused)
