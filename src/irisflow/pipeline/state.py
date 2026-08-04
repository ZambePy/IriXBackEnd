"""Pipeline state machine (SPRINTS.md §4.3).

Transitions publish :class:`~irisflow.core.events.StateChanged` on the
bus so any sink (cursor, telemetry, WebSocket) can react to state
changes without polling the pipeline object.

Valid transitions:

* ``IDLE ↔ CALIBRATING`` — Sprint 8.
* ``CALIBRATING → TRACKING`` — after a profile fits.
* ``TRACKING ↔ LOST`` — face lost / reacquired (hysteresis inside the
  tracker, not here).
* ``TRACKING ↔ PAUSED`` — Sprint 10 (dwell/safety).
* ``* → IDLE`` — ``stop()``.

The machine rejects impossible transitions (e.g. ``PAUSED → CALIBRATING``)
with a :class:`ValueError` so mis-wired callers fail loud.
"""

from __future__ import annotations

from irisflow.core.clock import Clock, SystemClock
from irisflow.core.events import PipelineState, StateChanged
from irisflow.logging import get_logger
from irisflow.pipeline.bus import EventBus

__all__ = ["PipelineStateMachine"]


_VALID_TRANSITIONS: dict[PipelineState, frozenset[PipelineState]] = {
    PipelineState.IDLE: frozenset({PipelineState.CALIBRATING, PipelineState.TRACKING}),
    PipelineState.CALIBRATING: frozenset({PipelineState.TRACKING, PipelineState.IDLE}),
    PipelineState.TRACKING: frozenset(
        {PipelineState.LOST, PipelineState.PAUSED, PipelineState.IDLE}
    ),
    PipelineState.LOST: frozenset({PipelineState.TRACKING, PipelineState.IDLE}),
    PipelineState.PAUSED: frozenset({PipelineState.TRACKING, PipelineState.IDLE}),
}


class PipelineStateMachine:
    """Track pipeline state and emit :class:`StateChanged` on every hop."""

    __slots__ = ("_bus", "_clock", "_log", "_state")

    def __init__(
        self,
        bus: EventBus,
        *,
        clock: Clock | None = None,
        initial: PipelineState = PipelineState.IDLE,
    ) -> None:
        self._bus = bus
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._state = initial
        self._log = get_logger("irisflow.pipeline.state")

    # ------------------------------------------------------------------ API
    @property
    def state(self) -> PipelineState:
        return self._state

    def transition_to(self, target: PipelineState) -> None:
        """Move to ``target`` if allowed. No-op when already there."""
        if target == self._state:
            return
        allowed = _VALID_TRANSITIONS[self._state]
        if target not in allowed:
            raise ValueError(
                f"Illegal transition {self._state.value} -> {target.value}. "
                f"Allowed from {self._state.value}: "
                f"{sorted(s.value for s in allowed)}"
            )
        previous = self._state
        self._state = target
        self._log.info(
            "state.transition",
            previous=previous.value,
            current=target.value,
        )
        self._bus.publish(
            StateChanged(
                previous=previous,
                current=target,
                timestamp=self._clock.monotonic(),
            )
        )
