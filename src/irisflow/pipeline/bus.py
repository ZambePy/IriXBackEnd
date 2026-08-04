"""Fan-out event bus — the seam between the pipeline and its consumers.

Synchronous, type-keyed, thread-unsafe (the pipeline publishes from one
thread). Consumers subscribe by event type and receive a callback; a
misbehaving handler is caught, logged, and does not affect the pipeline
or any other subscriber (rule #4 in SPRINTS.md §11.3: the bus never
takes the pipeline down).

For Sprint 12 (WebSocket), the same interface is enough — the WS sink
just becomes another subscriber that hands events off to an async queue.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from irisflow.core.events import Event
from irisflow.logging import get_logger

__all__ = ["EventBus", "EventHandler"]


EventHandler = Callable[[Event], None]


_WILDCARD = object()


class EventBus:
    """Simple synchronous pub/sub over :class:`~irisflow.core.events.Event`.

    Handlers may register for either a specific event type or every event
    (:meth:`subscribe_all`) — the latter is what the telemetry sink uses.
    """

    __slots__ = ("_handlers", "_log")

    def __init__(self) -> None:
        self._handlers: dict[Any, list[EventHandler]] = {}
        self._log = get_logger("irisflow.pipeline.bus")

    # ------------------------------------------------------------------ API
    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        """Register ``handler`` for events that are instances of ``event_type``."""
        self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register a handler that runs on every published event."""
        self._handlers.setdefault(_WILDCARD, []).append(handler)

    def unsubscribe(self, event_type: type, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            self._handlers.pop(event_type, None)

    def publish(self, event: Event) -> None:
        """Dispatch ``event`` to matching handlers plus every wildcard sink."""
        for handler in list(self._handlers.get(type(event), [])):
            self._dispatch(handler, event)
        for handler in list(self._handlers.get(_WILDCARD, [])):
            self._dispatch(handler, event)

    def clear(self) -> None:
        self._handlers.clear()

    def subscriber_count(self, event_type: type | None = None) -> int:
        if event_type is None:
            return sum(len(v) for v in self._handlers.values())
        return len(self._handlers.get(event_type, []))

    # ------------------------------------------------------------------ internals
    def _dispatch(self, handler: EventHandler, event: Event) -> None:
        try:
            handler(event)
        except Exception as exc:
            self._log.warning(
                "bus.handler_raised",
                handler=getattr(handler, "__qualname__", repr(handler)),
                event_type=type(event).__name__,
                error=str(exc),
            )
