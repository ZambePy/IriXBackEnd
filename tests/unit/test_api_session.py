"""Sprint 12 — :class:`ClientSession` + :class:`SessionHub` unit tests.

Focused on the backpressure contract from SPRINTS §11.3 rule #1: the
gaze queue drops the oldest on overflow, and a slow client can never
back-propagate into the pipeline thread.
"""

from __future__ import annotations

import asyncio

import pytest

from irisflow.api.schemas import GazeMessage, StateMessage
from irisflow.api.session import (
    DEFAULT_EVENT_QUEUE_SIZE,
    DEFAULT_GAZE_QUEUE_SIZE,
    ClientSession,
    SessionHub,
    translate_event,
)
from irisflow.core.events import GazeUpdated, PipelineState, StateChanged
from irisflow.pipeline.bus import EventBus


def _make_gaze(frame_id: int) -> GazeMessage:
    return GazeMessage(
        frame_id=frame_id,
        ts=float(frame_id),
        px=frame_id,
        py=frame_id,
        nx=0.1,
        ny=0.1,
        fixation=False,
        confidence=1.0,
    )


def _make_state() -> StateMessage:
    return StateMessage(ts=0.0, state="TRACKING", previous="IDLE")


@pytest.mark.asyncio
async def test_gaze_queue_drops_oldest_when_full() -> None:
    loop = asyncio.get_running_loop()
    session = ClientSession(loop=loop, gaze_queue_size=2)
    for i in range(5):
        session.offer(_make_gaze(i))
    assert session.gaze_queue.qsize() == 2
    assert session.dropped_gaze == 3
    # The surviving items are the two most recent.
    remaining = [session.gaze_queue.get_nowait(), session.gaze_queue.get_nowait()]
    assert [m.frame_id for m in remaining] == [3, 4]


@pytest.mark.asyncio
async def test_events_are_preferred_over_gaze_when_both_available() -> None:
    loop = asyncio.get_running_loop()
    session = ClientSession(loop=loop)
    session.offer(_make_gaze(1))
    session.offer(_make_state())
    first = await session.next_outgoing()
    assert isinstance(first, StateMessage)
    second = await session.next_outgoing()
    assert isinstance(second, GazeMessage)


def test_translate_event_covers_gaze_and_state() -> None:
    gaze = GazeUpdated(
        frame_id=7, timestamp=0.5, px=500, py=400,
        is_fixation=True, confidence=0.8,
    )
    msg = translate_event(gaze, screen_width=1000, screen_height=800)
    assert isinstance(msg, GazeMessage)
    assert msg.nx == 0.5
    assert msg.ny == 0.5
    assert msg.fixation is True

    state_change = StateChanged(
        previous=PipelineState.IDLE, current=PipelineState.TRACKING,
        timestamp=0.0,
    )
    msg2 = translate_event(state_change, screen_width=1000, screen_height=800)
    assert isinstance(msg2, StateMessage)
    assert msg2.state == "TRACKING"


@pytest.mark.asyncio
async def test_session_hub_dispatches_gaze_to_every_registered_client() -> None:
    loop = asyncio.get_running_loop()
    bus = EventBus()
    hub = SessionHub(bus, screen_width=1000, screen_height=800)
    hub.start()
    s1 = ClientSession(loop=loop)
    s2 = ClientSession(loop=loop)
    hub.register(s1)
    hub.register(s2)

    bus.publish(GazeUpdated(
        frame_id=1, timestamp=0.0, px=100, py=200,
        is_fixation=False, confidence=1.0,
    ))
    # Give the loop a tick to process call_soon_threadsafe scheduling.
    await asyncio.sleep(0)
    for session in (s1, s2):
        msg = session.gaze_queue.get_nowait()
        assert isinstance(msg, GazeMessage)
        assert msg.px == 100
        assert msg.py == 200


@pytest.mark.asyncio
async def test_defaults_match_sanity_bounds() -> None:
    assert DEFAULT_GAZE_QUEUE_SIZE >= 1
    assert DEFAULT_EVENT_QUEUE_SIZE >= DEFAULT_GAZE_QUEUE_SIZE
