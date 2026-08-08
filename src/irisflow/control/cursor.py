"""Pynput-backed cursor controller with a movement rate limiter.

Only touches the OS when :meth:`is_enabled` is True. Every entry point
guards on that flag, so a mis-wired sink can never nudge the mouse of a
developer who forgot to pass ``--cursor``.

Import strategy: pynput is imported lazily inside :meth:`_ensure_backend`
so that importing this module (e.g. from tests) never requires a display.
CI machines without an X server / Windows session can still import the
package.
"""

from __future__ import annotations

from typing import Any

from irisflow.core.clock import Clock, SystemClock
from irisflow.core.exceptions import IrisFlowError
from irisflow.logging import get_logger

__all__ = ["PynputCursor"]


class PynputCursor:
    """Moves and clicks the OS cursor via :mod:`pynput`, rate-limited.

    Args:
        rate_limit_hz: Maximum number of :meth:`move` calls per second
            that actually reach the OS. Extra calls are dropped
            silently; :meth:`is_enabled` still reports True.
        enabled: Whether the controller starts enabled. Even when False,
            :meth:`move` / :meth:`click` return without touching the OS.
        clock: Injected time source (mostly for tests).
    """

    __slots__ = (
        "_backend",
        "_button_type",
        "_clock",
        "_enabled",
        "_last_move_s",
        "_log",
        "_min_interval_s",
    )

    def __init__(
        self,
        *,
        rate_limit_hz: int = 60,
        enabled: bool = False,
        clock: Clock | None = None,
    ) -> None:
        if rate_limit_hz <= 0:
            raise ValueError(f"rate_limit_hz must be > 0, got {rate_limit_hz}")
        self._min_interval_s = 1.0 / rate_limit_hz
        self._enabled = enabled
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._last_move_s: float | None = None
        self._log = get_logger("irisflow.control.cursor")
        self._backend: Any | None = None  # pynput.mouse.Controller instance
        self._button_type: Any | None = None  # pynput.mouse.Button enum

    # ------------------------------------------------------------------ API
    def enable(self) -> None:
        if not self._enabled:
            self._log.info("cursor.enabled", backend="pynput")
        self._enabled = True

    def disable(self) -> None:
        if self._enabled:
            self._log.info("cursor.disabled", backend="pynput")
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def move(self, px: int, py: int) -> None:
        if not self._enabled:
            return
        now = self._clock.monotonic()
        if (
            self._last_move_s is not None
            and (now - self._last_move_s) < self._min_interval_s
        ):
            return
        backend = self._ensure_backend()
        try:
            backend.position = (int(px), int(py))
        except Exception as exc:
            self._log.warning("cursor.move_failed", error=str(exc))
            return
        self._last_move_s = now

    def click(self, button: str = "left") -> None:
        if not self._enabled:
            return
        backend = self._ensure_backend()
        button_enum = self._resolve_button(button)
        try:
            backend.click(button_enum, 1)
        except Exception as exc:
            self._log.warning("cursor.click_failed", error=str(exc), button=button)

    # ------------------------------------------------------------------ internals
    def _ensure_backend(self) -> Any:
        if self._backend is None:
            try:
                from pynput import mouse
            except Exception as exc:  # pragma: no cover - only when pynput missing
                raise IrisFlowError(
                    "pynput is required for PynputCursor but failed to import. "
                    f"Install it (already a base dep) or use NoOpCursor. ({exc})"
                ) from exc
            self._backend = mouse.Controller()
            self._button_type = mouse.Button
        return self._backend

    def _resolve_button(self, button: str) -> Any:
        self._ensure_backend()
        button_enum = self._button_type
        assert button_enum is not None
        try:
            return getattr(button_enum, button)
        except AttributeError as exc:
            raise IrisFlowError(
                f"unknown mouse button {button!r}; expected left|right|middle"
            ) from exc
