"""Bridge between the (thread-driven) pipeline and the (asyncio) WS clients.

Design constraints from SPRINTS §11.3:

1. **Backpressure by discard.** Every client has a small ``asyncio.Queue``.
   High-frequency gaze messages are dropped when the queue fills; only
   the most recent one survives. Never block the pipeline thread on a
   slow socket.
2. **Separation of channels.** Gaze positions are transient; state and
   calibration events must be delivered. We keep them in the same queue
   for simplicity but the drop policy protects state events by size
   (state events are rare so they rarely reach the cap).
3. **Thread → asyncio bridge.** The bus fires on the pipeline thread; we
   hand messages to the asyncio loop via
   :meth:`asyncio.AbstractEventLoop.call_soon_threadsafe`.
4. The API layer never touches tracking logic — only translation.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from irisflow.api.schemas import (
    CalibrationPointMessage,
    ClickMessage,
    DwellProgressMessage,
    ErrorMessage,
    FaceLostMessage,
    GazeMessage,
    PipelineReadyMessage,
    SafetyMessage,
    ServerMessage,
    StateMessage,
)
from irisflow.core.events import (
    CalibrationProgress,
    DwellClick,
    DwellProgress,
    Event,
    FaceLost,
    GazeUpdated,
    RawGazeReady,
    SafetyPaused,
    SafetyResumed,
    StateChanged,
)

if TYPE_CHECKING:
    from irisflow.pipeline.bus import EventBus

__all__ = [
    "ClientSession",
    "SessionHub",
    "translate_event",
]


DEFAULT_GAZE_QUEUE_SIZE = 4
DEFAULT_EVENT_QUEUE_SIZE = 64


def translate_event(event: Event, *, screen_width: int, screen_height: int) -> ServerMessage | None:
    """Map a bus :class:`Event` to a WS :class:`ServerMessage`.

    Returns ``None`` for events without a wire counterpart (e.g. the
    :class:`~irisflow.core.events.RawGazeReady` internal signal that the
    WebSocket clients don't need).
    """
    if isinstance(event, GazeUpdated):
        nx = event.px / screen_width if screen_width > 0 else 0.0
        ny = event.py / screen_height if screen_height > 0 else 0.0
        return GazeMessage(
            frame_id=event.frame_id,
            ts=event.timestamp,
            px=event.px,
            py=event.py,
            nx=nx,
            ny=ny,
            fixation=event.is_fixation,
            confidence=event.confidence,
        )
    if isinstance(event, StateChanged):
        return StateMessage(
            ts=event.timestamp,
            state=event.current.value,
            previous=event.previous.value,
        )
    if isinstance(event, FaceLost):
        return FaceLostMessage(ts=event.timestamp, duration_ms=event.duration_ms)
    if isinstance(event, DwellProgress):
        return DwellProgressMessage(
            ts=event.timestamp,
            frame_id=event.frame_id,
            px=event.px,
            py=event.py,
            progress=event.progress,
            radius_px=event.radius_px,
        )
    if isinstance(event, DwellClick):
        return ClickMessage(
            ts=event.timestamp,
            frame_id=event.frame_id,
            px=event.px,
            py=event.py,
            button=event.button,
        )
    if isinstance(event, CalibrationProgress):
        return CalibrationPointMessage(
            ts=0.0,  # calibration events don't carry a bus timestamp
            index=event.index,
            total=event.total,
            x=event.target_x,
            y=event.target_y,
            phase=event.phase,
        )
    if isinstance(event, SafetyPaused):
        return SafetyMessage(ts=event.timestamp, paused=True, reason=event.reason)
    if isinstance(event, SafetyResumed):
        return SafetyMessage(ts=event.timestamp, paused=False, reason=None)
    return None


@dataclass
class ClientSession:
    """One connected WebSocket client.

    Two separate queues keep transient gaze from crowding out state /
    calibration events: gaze uses a very small queue with drop-oldest,
    while events use a larger queue that only overflows in pathological
    cases.
    """

    loop: asyncio.AbstractEventLoop
    gaze_queue: asyncio.Queue[GazeMessage] = field(init=False)
    event_queue: asyncio.Queue[ServerMessage] = field(init=False)
    gaze_queue_size: int = DEFAULT_GAZE_QUEUE_SIZE
    event_queue_size: int = DEFAULT_EVENT_QUEUE_SIZE
    dropped_gaze: int = 0
    dropped_events: int = 0

    def __post_init__(self) -> None:
        self.gaze_queue = asyncio.Queue(maxsize=self.gaze_queue_size)
        self.event_queue = asyncio.Queue(maxsize=self.event_queue_size)

    def offer(self, message: ServerMessage) -> None:
        """Enqueue ``message``, dropping the oldest on overflow.

        Called from the asyncio loop only. The thread → loop hop happens
        in :meth:`SessionHub._dispatch_message`.
        """
        if isinstance(message, GazeMessage):
            if self.gaze_queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.gaze_queue.get_nowait()
                self.dropped_gaze += 1
            self.gaze_queue.put_nowait(message)
        else:
            if self.event_queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.event_queue.get_nowait()
                self.dropped_events += 1
            self.event_queue.put_nowait(message)

    async def next_outgoing(self) -> ServerMessage:
        """Await the next message to send, preferring event queue.

        Events beat gaze so state changes and calibration progress are
        never starved by 30 Hz gaze traffic. When both queues have items
        the gaze one stays in the queue (rather than being discarded)
        for the next call.
        """
        # Fast path: an event is already waiting — take it and skip the
        # asyncio.wait dance below.
        if not self.event_queue.empty():
            return self.event_queue.get_nowait()
        event_task = asyncio.create_task(self.event_queue.get())
        gaze_task = asyncio.create_task(self.gaze_queue.get())
        try:
            done, pending = await asyncio.wait(
                {event_task, gaze_task}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            event_task.cancel()
            gaze_task.cancel()
            raise
        for task in pending:
            task.cancel()
        # If we happened to pull a gaze message but the caller wants events
        # first, push the gaze back so it isn't lost.
        if event_task in done and gaze_task in done:
            gaze_msg = gaze_task.result()
            try:
                self.gaze_queue.put_nowait(gaze_msg)
            except asyncio.QueueFull:
                self.dropped_gaze += 1
            return event_task.result()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if event_task in done:
            return event_task.result()
        return gaze_task.result()


class SessionHub:
    """Fan-out from the pipeline event bus to every connected WS client.

    A single hub is created per FastAPI app; every WS handshake calls
    :meth:`register` on connect and :meth:`unregister` on disconnect. The
    hub subscribes to the bus once via :meth:`start`.
    """

    __slots__ = (
        "_bus",
        "_lock",
        "_ready_sent",
        "_screen_height",
        "_screen_width",
        "_sessions",
        "_started",
    )

    def __init__(
        self,
        bus: EventBus,
        *,
        screen_width: int,
        screen_height: int,
    ) -> None:
        self._bus = bus
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._sessions: list[ClientSession] = []
        self._lock = threading.Lock()
        self._started = False
        self._ready_sent: bool = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._started:
            return
        self._bus.subscribe_all(self._on_event)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._bus.clear()  # tolerant: safer than tracking every subscription
        self._started = False

    def update_screen(self, *, screen_width: int, screen_height: int) -> None:
        """Refresh normalized-coordinate divisor when the client tells us the resolution."""
        self._screen_width = screen_width
        self._screen_height = screen_height

    # ------------------------------------------------------------------ sessions
    def register(self, session: ClientSession) -> None:
        with self._lock:
            self._sessions.append(session)

    def unregister(self, session: ClientSession) -> None:
        with self._lock:
            try:
                self._sessions.remove(session)
            except ValueError:
                return

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------ dispatch
    def _on_event(self, event: Event) -> None:
        """Bus callback — runs on the pipeline thread."""
        import time as _time
        if not self._ready_sent and isinstance(event, RawGazeReady):
            self._ready_sent = True
            ready_msg = PipelineReadyMessage(ts=_time.monotonic())
            with self._lock:
                targets = list(self._sessions)
            for session in targets:
                self._dispatch_message(session, ready_msg)

        message = translate_event(
            event,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
        )
        if message is None:
            return
        with self._lock:
            targets = list(self._sessions)
        for session in targets:
            self._dispatch_message(session, message)

    def _dispatch_message(self, session: ClientSession, message: ServerMessage) -> None:
        loop = session.loop
        try:
            loop.call_soon_threadsafe(session.offer, message)
        except RuntimeError:
            # loop closed while a stale session was still in our list —
            # unregister it so we don't keep trying.
            self.unregister(session)


def make_error(code: str, message: str) -> ErrorMessage:
    """Shorthand for building an :class:`ErrorMessage`."""
    return ErrorMessage(code=code, message=message)


PublisherCallable = Callable[[Event], None]
"""Type alias matching :meth:`EventBus.publish`."""
