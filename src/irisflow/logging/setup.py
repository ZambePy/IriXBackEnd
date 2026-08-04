"""structlog wiring for IrisFlow.

Two sinks live side by side:

* **Human console** — colorful, one line per event, meant for the developer
  running ``irisflow`` in a terminal.
* **JSON file** — one JSON object per line in ``data/logs/`` (rotated by the
  caller if needed). This is the machine-readable trail used by the telemetry
  layer and by post-mortem debugging of a real user's session.

Every event carries at least a monotonic ``timestamp`` and a ``level``. The
loop stage that produced the event is expected to bind a ``frame_id`` (and
often ``stage``) via :func:`structlog.contextvars.bind_contextvars` so that
each log entry can be traced back to the exact frame it came from — this is
what makes latency/quality debugging tractable once we hit Sprint 7.

The public surface is intentionally tiny:

>>> configure_logging(app_config.logging)
>>> log = get_logger(__name__)
>>> with bound(frame_id=42, stage="inference"):
...     log.info("gaze.raw", x=0.5, y=0.5, inference_ms=12.3)
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
)
from structlog.typing import EventDict, Processor

__all__ = [
    "bind",
    "bind_frame",
    "bound",
    "clear_context",
    "configure_logging",
    "get_logger",
]


_LEVEL_TO_INT: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(
    *,
    level: str = "INFO",
    human_console: bool = True,
    json_file: Path | None = None,
    console_stream: TextIO | None = None,
) -> None:
    """Configure structlog + stdlib logging with the given sinks.

    Safe to call multiple times: each call fully replaces the previous
    configuration. Tests use this to swap sinks in and out.

    Primitives instead of a config object keep the ``logging`` package free of
    dependencies on ``config`` — both live in the same architectural layer.
    The CLI/orchestration layer is responsible for the ``LoggingConfig →
    configure_logging`` translation.

    Args:
        level: One of ``DEBUG|INFO|WARNING|ERROR|CRITICAL``.
        human_console: When true, write colored one-line events to stderr.
        json_file: When set, append one JSON object per event to this path.
        console_stream: Overrides ``sys.stderr`` — used by tests to capture
            output without touching the real stderr.

    Raises:
        ValueError: ``level`` is not a recognised name.
    """
    if level not in _LEVEL_TO_INT:
        raise ValueError(
            f"Unknown log level {level!r}; expected one of {sorted(_LEVEL_TO_INT)}"
        )
    level_int = _LEVEL_TO_INT[level]

    handlers: list[logging.Handler] = []

    if human_console:
        console_handler = logging.StreamHandler(console_stream or sys.stderr)
        console_handler.setFormatter(
            _stdlib_formatter(
                structlog.dev.ConsoleRenderer(
                    colors=console_stream is None,
                    exception_formatter=structlog.dev.plain_traceback,
                ),
            ),
        )
        handlers.append(console_handler)

    if json_file is not None:
        _ensure_parent(json_file)
        file_handler = logging.FileHandler(json_file, encoding="utf-8")
        file_handler.setFormatter(
            _stdlib_formatter(structlog.processors.JSONRenderer(sort_keys=True)),
        )
        handlers.append(file_handler)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        handler.setLevel(level_int)
        root.addHandler(handler)
    root.setLevel(level_int)

    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound logger; use the module name (``__name__``) as ``name``.

    Typed as :data:`~typing.Any` because structlog's ``BoundLogger`` protocol
    doesn't play well with strict type checkers — callers use it as a duck-typed
    logger with ``debug/info/warning/error/critical``.
    """
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind(**values: Any) -> None:
    """Bind values into the ambient context for the current async/thread scope."""
    bind_contextvars(**values)


def bind_frame(frame_id: int, **extra: Any) -> None:
    """Convenience wrapper: bind ``frame_id`` (and any siblings) at once."""
    bind_contextvars(frame_id=frame_id, **extra)


def clear_context() -> None:
    """Drop every bound context variable — call between frames if needed."""
    clear_contextvars()


@contextmanager
def bound(**values: Any) -> Iterator[None]:
    """Bind ``values`` for the duration of a ``with`` block, then unbind."""
    bind_contextvars(**values)
    try:
        yield
    finally:
        unbind_contextvars(*values.keys())


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _shared_processors() -> list[Processor]:
    """Processors applied before formatting by every sink."""
    return [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_monotonic_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def _stdlib_formatter(renderer: Processor) -> structlog.stdlib.ProcessorFormatter:
    """Wrap a final renderer so records from stdlib and structlog share it."""
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def _add_monotonic_timestamp(
    _logger: Any, _method: str, event_dict: EventDict
) -> EventDict:
    """Attach both a monotonic ``ts_mono`` and an ISO ``timestamp``.

    Monotonic time is what latency measurements are computed against; wall
    time is what humans read. Keeping both avoids re-deriving one from the
    other later and being off by a leap-second or a suspend/resume.
    """
    import time
    from datetime import UTC, datetime

    event_dict.setdefault("ts_mono", time.monotonic())
    event_dict.setdefault("timestamp", datetime.now(UTC).isoformat(timespec="milliseconds"))
    return event_dict


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
