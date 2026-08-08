"""Bus subscriber that records a session to disk (Sprint 11).

The recorder writes the JSONL session format defined in :mod:`session`.
Its bus contract is intentionally narrow — it only observes events and
never mutates state or throws back into the pipeline. Any write error
is caught, logged, and the pipeline keeps running.

Threading: the pipeline publishes events synchronously from a single
worker thread, so the recorder needs no lock. If a future async sink
lands, it will subscribe on its own thread; the recorder itself is
still safe because :meth:`start` / :meth:`stop` are always called from
the owning thread.

Overhead budget (SPRINTS §11): serialising ~10 short floats and
writing one line per frame at 30 Hz costs well under the 3 % of frame
time budget on any local disk. The recorder buffers nothing beyond the
standard file object's write cache.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import structlog

from irisflow.core.clock import Clock, SystemClock
from irisflow.core.events import (
    CalibrationProgress,
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
from irisflow.telemetry.session import (
    SessionHeader,
    event_to_line,
    header_to_line,
)

__all__ = ["SessionRecorder"]


_RECORDED_TYPES: tuple[type, ...] = (
    RawGazeReady,
    GazeUpdated,
    FaceLost,
    FaceAcquired,
    StateChanged,
    CalibrationProgress,
    DwellProgress,
    DwellClick,
    SafetyPaused,
    SafetyResumed,
)


class SessionRecorder:
    """Subscribe to a bus and stream every relevant event to a JSONL file.

    Args:
        output_path: File to write. Parent directory is created on
            :meth:`start` if missing.
        header: Session header written once as the first line.
        clock: Time source used for the ``wall_start_s`` fallback and
            any future timestamping. Mostly injected for tests.
        on_write_error: Optional callback invoked when a write fails.
            Defaults to logging a warning and swallowing.
    """

    __slots__ = (
        "_clock",
        "_events_written",
        "_file",
        "_header",
        "_log",
        "_on_write_error",
        "_output_path",
        "_started",
        "_subscribe",
        "_unsubscribe",
    )

    def __init__(
        self,
        *,
        output_path: Path,
        header: SessionHeader,
        subscribe: Callable[[type, Callable[[Event], None]], None],
        unsubscribe: Callable[[type, Callable[[Event], None]], None],
        clock: Clock | None = None,
        on_write_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._output_path = output_path
        self._header = header
        self._subscribe = subscribe
        self._unsubscribe = unsubscribe
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._on_write_error = on_write_error
        self._file: IO[str] | None = None
        self._started = False
        self._events_written = 0
        self._log = structlog.get_logger("irisflow.telemetry.recorder")

    # ------------------------------------------------------------------ API
    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def events_written(self) -> int:
        return self._events_written

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Open the file, write the header, subscribe to every bus event."""
        if self._started:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._output_path.open("w", encoding="utf-8", newline="\n")
        self._file.write(header_to_line(self._header))
        self._file.write("\n")
        for event_type in _RECORDED_TYPES:
            self._subscribe(event_type, self._on_event)
        self._started = True
        self._log.info(
            "session.record_started",
            path=str(self._output_path),
            session_id=self._header.session_id,
        )

    def stop(self) -> None:
        """Unsubscribe, flush and close. Idempotent."""
        if not self._started:
            return
        for event_type in _RECORDED_TYPES:
            self._unsubscribe(event_type, self._on_event)
        try:
            if self._file is not None:
                self._file.flush()
                self._file.close()
        except Exception as exc:
            self._log.warning("session.close_failed", error=str(exc))
        self._file = None
        self._started = False
        self._log.info(
            "session.record_stopped",
            path=str(self._output_path),
            events=self._events_written,
        )

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> SessionRecorder:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ handler
    def _on_event(self, event: Event) -> None:
        if not self._started or self._file is None:
            return
        try:
            self._file.write(event_to_line(event))
            self._file.write("\n")
            self._events_written += 1
        except Exception as exc:  # pragma: no cover - disk full etc.
            self._log.warning(
                "session.write_failed",
                error=str(exc),
                event_type=type(event).__name__,
            )
            if self._on_write_error is not None:
                self._on_write_error(exc)
