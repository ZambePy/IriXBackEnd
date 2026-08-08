"""Production no-op cursor controller.

Used by the CLI ``irisflow run --no-cursor`` (the default) and by
integration tests that need a real production-side cursor sink without
touching the OS. Distinct from :class:`tests.fixtures.stubs.NoOpCursorController`
which lives under ``tests/`` — the import-linter contract in
``pyproject.toml`` forbids production code from importing that test stub.
"""

from __future__ import annotations

from irisflow.logging import get_logger

__all__ = ["NoOpCursor"]


class NoOpCursor:
    """Records nothing, moves nothing, clicks nothing.

    Implements :class:`~irisflow.core.interfaces.CursorController` structurally.
    Enable/disable state is honoured so downstream code can react to
    ``is_enabled`` without special-casing this class.
    """

    __slots__ = ("_enabled", "_log")

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._log = get_logger("irisflow.control.noop")

    def move(self, px: int, py: int) -> None:
        return None

    def click(self, button: str = "left") -> None:
        return None

    def enable(self) -> None:
        if not self._enabled:
            self._log.info("cursor.enabled", backend="noop")
        self._enabled = True

    def disable(self) -> None:
        if self._enabled:
            self._log.info("cursor.disabled", backend="noop")
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled
