"""Shared plumbing for every :class:`FrameSource` in the capture layer.

The two upsides of putting the common bits here:

* Concrete sources (webcam, video file, synthetic) only declare what makes
  them different — everyone gets the same clock injection, frame-id counter
  and context-manager semantics without cut-and-paste.
* Tests can spin up a source with a :class:`~irisflow.core.clock.FakeClock`
  so ``Frame.timestamp`` is deterministic across the whole suite.
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

from irisflow.core.clock import Clock, SystemClock

__all__ = ["BaseFrameSource"]


class BaseFrameSource:
    """Non-abstract mixin: subclasses override :meth:`_do_open`, :meth:`_do_close`
    and :meth:`read`. Kept as a plain class (not an ABC) so mypy strict is happy
    and structural :class:`~irisflow.core.interfaces.FrameSource` conformance is
    verified by tests, not by inheritance.

    Args:
        clock: Time source. Injected so tests can freeze time — never call
            :func:`time.monotonic` inside a source, always ``self._clock``.
    """

    __slots__ = ("_clock", "_frame_id", "_open")

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._frame_id: int = 0
        self._open: bool = False

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> None:
        """Idempotent open. Subclasses override :meth:`_do_open`."""
        if self._open:
            return
        self._do_open()
        self._open = True

    def close(self) -> None:
        """Idempotent close. Subclasses override :meth:`_do_close`."""
        if not self._open:
            return
        try:
            self._do_close()
        finally:
            self._open = False

    def _do_open(self) -> None:
        """Subclass hook: acquire underlying resource (camera, file handle)."""

    def _do_close(self) -> None:
        """Subclass hook: release underlying resource."""

    # ------------------------------------------------------------------ state
    @property
    def is_open(self) -> bool:
        return self._open

    # ------------------------------------------------------------------ ids
    def _next_frame_id(self) -> int:
        """Return the next monotonically increasing frame id."""
        frame_id = self._frame_id
        self._frame_id += 1
        return frame_id

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
