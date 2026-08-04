"""Sprint 7 — synchronous pub/sub event bus."""

from __future__ import annotations

from irisflow.core.events import FaceLost, PipelineState, StateChanged
from irisflow.pipeline.bus import EventBus


def _face_lost(frame_id: int = 0) -> FaceLost:
    return FaceLost(frame_id=frame_id, timestamp=0.0, duration_ms=0.0)


def _state_changed() -> StateChanged:
    return StateChanged(
        previous=PipelineState.IDLE, current=PipelineState.TRACKING, timestamp=0.0
    )


def test_typed_subscription_only_receives_matching_events() -> None:
    bus = EventBus()
    received: list = []
    bus.subscribe(FaceLost, received.append)

    bus.publish(_face_lost())
    bus.publish(_state_changed())

    assert len(received) == 1
    assert isinstance(received[0], FaceLost)


def test_subscribe_all_receives_every_event() -> None:
    bus = EventBus()
    all_events: list = []
    bus.subscribe_all(all_events.append)

    bus.publish(_face_lost())
    bus.publish(_state_changed())

    assert len(all_events) == 2


def test_multiple_handlers_all_fire_on_publish() -> None:
    bus = EventBus()
    a: list = []
    b: list = []
    bus.subscribe(FaceLost, a.append)
    bus.subscribe(FaceLost, b.append)
    bus.publish(_face_lost())
    assert len(a) == 1
    assert len(b) == 1


def test_unsubscribe_removes_only_the_specific_handler() -> None:
    bus = EventBus()
    a: list = []
    b: list = []
    bus.subscribe(FaceLost, a.append)
    bus.subscribe(FaceLost, b.append)
    bus.unsubscribe(FaceLost, a.append)
    bus.publish(_face_lost())
    assert a == []
    assert len(b) == 1


def test_handler_exception_does_not_break_the_bus() -> None:
    bus = EventBus()
    survivor: list = []

    def _boom(_: object) -> None:
        raise RuntimeError("bus should absorb this")

    bus.subscribe(FaceLost, _boom)
    bus.subscribe(FaceLost, survivor.append)
    bus.publish(_face_lost())
    assert len(survivor) == 1


def test_clear_removes_all_handlers() -> None:
    bus = EventBus()
    bus.subscribe(FaceLost, lambda _: None)
    bus.subscribe_all(lambda _: None)
    assert bus.subscriber_count() == 2
    bus.clear()
    assert bus.subscriber_count() == 0
