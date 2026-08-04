"""Sprint 7 — pipeline state machine + StateChanged events."""

from __future__ import annotations

import pytest

from irisflow.core.clock import FakeClock
from irisflow.core.events import PipelineState, StateChanged
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.state import PipelineStateMachine


def test_default_state_is_idle() -> None:
    machine = PipelineStateMachine(EventBus())
    assert machine.state == PipelineState.IDLE


def test_transition_publishes_state_changed() -> None:
    bus = EventBus()
    clock = FakeClock()
    events: list = []
    bus.subscribe(StateChanged, events.append)

    machine = PipelineStateMachine(bus, clock=clock)
    clock.advance(1.5)
    machine.transition_to(PipelineState.TRACKING)

    assert machine.state == PipelineState.TRACKING
    assert len(events) == 1
    assert events[0].previous == PipelineState.IDLE
    assert events[0].current == PipelineState.TRACKING
    assert events[0].timestamp == 1.5


def test_transition_to_current_state_is_noop() -> None:
    bus = EventBus()
    events: list = []
    bus.subscribe(StateChanged, events.append)
    machine = PipelineStateMachine(bus, initial=PipelineState.TRACKING)
    machine.transition_to(PipelineState.TRACKING)
    assert events == []


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (PipelineState.IDLE, PipelineState.LOST),
        (PipelineState.IDLE, PipelineState.PAUSED),
        (PipelineState.LOST, PipelineState.CALIBRATING),
        (PipelineState.PAUSED, PipelineState.CALIBRATING),
        (PipelineState.PAUSED, PipelineState.LOST),
    ],
)
def test_illegal_transitions_raise(
    start: PipelineState, target: PipelineState
) -> None:
    machine = PipelineStateMachine(EventBus(), initial=start)
    with pytest.raises(ValueError, match="Illegal transition"):
        machine.transition_to(target)


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (PipelineState.IDLE, PipelineState.TRACKING),
        (PipelineState.IDLE, PipelineState.CALIBRATING),
        (PipelineState.CALIBRATING, PipelineState.TRACKING),
        (PipelineState.TRACKING, PipelineState.LOST),
        (PipelineState.TRACKING, PipelineState.PAUSED),
        (PipelineState.LOST, PipelineState.TRACKING),
        (PipelineState.PAUSED, PipelineState.TRACKING),
        (PipelineState.TRACKING, PipelineState.IDLE),
        (PipelineState.LOST, PipelineState.IDLE),
    ],
)
def test_valid_transitions_succeed(
    start: PipelineState, target: PipelineState
) -> None:
    machine = PipelineStateMachine(EventBus(), initial=start)
    machine.transition_to(target)
    assert machine.state == target
