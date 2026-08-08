"""Global-hotkey listener that fires when the user hits the kill switch.

Uses :mod:`pynput.keyboard.GlobalHotKeys`. Kept in its own module because
importing pynput's keyboard listener starts a background thread on some
platforms — we only want that side effect when the CLI explicitly opts in.

Callers pass an ``on_trigger`` callback that typically:

* calls :meth:`~irisflow.control.safety.SafetyGate.trigger_pause("kill_switch")`;
* calls :meth:`~irisflow.core.interfaces.CursorController.disable`;
* publishes a :class:`~irisflow.core.events.SafetyPaused` event on the bus.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from irisflow.core.exceptions import IrisFlowError
from irisflow.logging import get_logger

__all__ = ["KillSwitchListener", "parse_hotkey"]


def parse_hotkey(spec: str) -> str:
    """Convert a user-facing hotkey (``"ctrl+alt+esc"``) to pynput syntax.

    pynput expects modifiers wrapped in angle brackets (``"<ctrl>+<alt>+<esc>"``)
    and single characters bare. Names we translate: ``ctrl``, ``alt``,
    ``shift``, ``cmd``, ``esc``, ``tab``, ``space``, ``enter``, function
    keys (``f1``..``f12``).
    """
    if not spec:
        raise IrisFlowError("kill switch hotkey must not be empty")
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise IrisFlowError(f"kill switch hotkey {spec!r} is malformed")
    named = {
        "ctrl", "alt", "shift", "cmd", "esc", "tab", "space", "enter", "return",
        "backspace", "delete", "insert", "home", "end", "page_up", "page_down",
        "up", "down", "left", "right",
    }
    rendered: list[str] = []
    for token in parts:
        if token in named or (token.startswith("f") and token[1:].isdigit()):
            rendered.append(f"<{token}>")
        elif len(token) == 1:
            rendered.append(token)
        else:
            raise IrisFlowError(f"kill switch hotkey token {token!r} is unknown")
    return "+".join(rendered)


class KillSwitchListener:
    """Owns a :class:`pynput.keyboard.GlobalHotKeys` listener thread.

    Not started on construction — call :meth:`start` explicitly from the
    CLI when the user opts into real cursor control. Tests can drive the
    same callback by calling :meth:`trigger` directly, bypassing pynput.
    """

    __slots__ = ("_hotkey_spec", "_listener", "_log", "_on_trigger", "_started")

    def __init__(self, *, hotkey: str, on_trigger: Callable[[], None]) -> None:
        self._hotkey_spec = parse_hotkey(hotkey)
        self._on_trigger = on_trigger
        self._log = get_logger("irisflow.control.kill_switch")
        self._listener: Any | None = None
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the pynput global-hotkey listener thread."""
        if self._started:
            return
        try:
            from pynput import keyboard
        except Exception as exc:  # pragma: no cover - only when pynput missing
            raise IrisFlowError(
                f"pynput.keyboard is required for the kill switch ({exc})"
            ) from exc
        try:
            listener = keyboard.GlobalHotKeys({self._hotkey_spec: self._safe_trigger})
            listener.start()
        except Exception as exc:  # pragma: no cover - platform-specific
            raise IrisFlowError(
                f"failed to start kill switch listener for {self._hotkey_spec!r}: {exc}"
            ) from exc
        self._listener = listener
        self._started = True
        self._log.info("kill_switch.started", hotkey=self._hotkey_spec)

    def stop(self) -> None:
        if self._listener is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - platform-specific
                self._listener.stop()
            self._listener = None
        self._started = False

    # ------------------------------------------------------------------ testing
    def trigger(self) -> None:
        """Directly invoke the callback — the hook tests use instead of pynput."""
        self._safe_trigger()

    @property
    def is_running(self) -> bool:
        return self._started

    # ------------------------------------------------------------------ internals
    def _safe_trigger(self) -> None:
        try:
            self._on_trigger()
        except Exception as exc:
            self._log.warning("kill_switch.callback_failed", error=str(exc))
