"""Injectable time source.

The pipeline never calls :func:`time.monotonic` or :func:`time.sleep`
directly — it holds a :class:`Clock` and calls it. Two upsides:

* Latency numbers are computed against a single, mockable clock.
* Filters and dwell logic become deterministic in tests via
  :class:`FakeClock.advance`, which is what makes the One Euro filter and
  the calibration collector unit-testable without wall-clock timing tricks.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "FakeClock", "SystemClock"]


@runtime_checkable
class Clock(Protocol):
    """Structural interface every clock in the system satisfies."""

    def monotonic(self) -> float:
        """Return seconds from an arbitrary monotonic origin."""
        ...

    def wall(self) -> float:
        """Return seconds since the Unix epoch. Only for logs, never for math."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``. ``FakeClock`` returns immediately."""
        ...


class SystemClock:
    """Real clock backed by :mod:`time`. The only :class:`Clock` used in prod."""

    __slots__ = ()

    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Deterministic clock for tests.

    ``sleep(seconds)`` never blocks — it advances the clock by ``seconds``.
    Tests that need to keep sleep and advancement separate can call
    :meth:`advance` explicitly and use ``sleep_blocks=False`` semantics by
    ignoring the sleep call in the code under test.
    """

    __slots__ = ("_now", "_wall")

    def __init__(self, start: float = 0.0, wall: float = 0.0) -> None:
        self._now = float(start)
        self._wall = float(wall)

    def monotonic(self) -> float:
        return self._now

    def wall(self) -> float:
        return self._wall

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"sleep seconds must be >= 0, got {seconds}")
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Move the monotonic (and wall) clock forward by ``seconds``."""
        if seconds < 0:
            raise ValueError(f"advance seconds must be >= 0, got {seconds}")
        self._now += float(seconds)
        self._wall += float(seconds)

    def set_wall(self, seconds: float) -> None:
        """Set the wall time independently — useful when a test wants a
        specific date without touching the monotonic clock."""
        self._wall = float(seconds)
