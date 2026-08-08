"""Sprint 10 — hotkey parser and KillSwitchListener trigger callback."""

from __future__ import annotations

import pytest

from irisflow.control.kill_switch import KillSwitchListener, parse_hotkey
from irisflow.core.exceptions import IrisFlowError


def test_parse_hotkey_wraps_named_keys() -> None:
    assert parse_hotkey("ctrl+alt+esc") == "<ctrl>+<alt>+<esc>"


def test_parse_hotkey_bare_char_stays_bare() -> None:
    assert parse_hotkey("ctrl+q") == "<ctrl>+q"


def test_parse_hotkey_supports_function_keys() -> None:
    assert parse_hotkey("ctrl+f9") == "<ctrl>+<f9>"


def test_parse_hotkey_rejects_empty() -> None:
    with pytest.raises(IrisFlowError):
        parse_hotkey("")


def test_parse_hotkey_rejects_unknown_token() -> None:
    with pytest.raises(IrisFlowError):
        parse_hotkey("ctrl+banana")


def test_listener_trigger_invokes_callback_without_starting_pynput() -> None:
    fired = {"n": 0}

    def _on_trigger() -> None:
        fired["n"] += 1

    listener = KillSwitchListener(hotkey="ctrl+alt+esc", on_trigger=_on_trigger)
    listener.trigger()
    listener.trigger()
    assert fired["n"] == 2
    assert not listener.is_running


def test_listener_swallows_callback_exception() -> None:
    def _raises() -> None:
        raise RuntimeError("boom")

    listener = KillSwitchListener(hotkey="ctrl+alt+esc", on_trigger=_raises)
    listener.trigger()  # must not propagate
