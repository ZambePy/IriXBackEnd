"""Sprint 10 — PynputCursor with a mocked pynput backend."""

from __future__ import annotations

import pytest

from irisflow.control.cursor import PynputCursor
from irisflow.core.clock import FakeClock
from irisflow.core.exceptions import IrisFlowError


class _FakeBackend:
    """Minimal stand-in for :class:`pynput.mouse.Controller`."""

    def __init__(self) -> None:
        self.position = (0, 0)
        self.clicks: list[tuple[object, int]] = []

    def click(self, button: object, count: int) -> None:
        self.clicks.append((button, count))


class _FakeButton:
    left = "LEFT"
    right = "RIGHT"
    middle = "MIDDLE"


def _install_backend(cursor: PynputCursor) -> _FakeBackend:
    backend = _FakeBackend()
    cursor._backend = backend
    cursor._button_type = _FakeButton
    return backend


def test_rate_limit_hz_must_be_positive() -> None:
    with pytest.raises(ValueError, match="rate_limit_hz"):
        PynputCursor(rate_limit_hz=0)


def test_disabled_cursor_never_touches_backend() -> None:
    cursor = PynputCursor(rate_limit_hz=60, enabled=False)
    backend = _install_backend(cursor)
    cursor.move(100, 100)
    cursor.click()
    assert backend.position == (0, 0)
    assert backend.clicks == []


def test_enable_disable_toggles_state() -> None:
    cursor = PynputCursor(rate_limit_hz=60)
    assert not cursor.is_enabled
    cursor.enable()
    assert cursor.is_enabled
    cursor.disable()
    assert not cursor.is_enabled


def test_move_is_rate_limited() -> None:
    clock = FakeClock()
    cursor = PynputCursor(rate_limit_hz=100, enabled=True, clock=clock)
    backend = _install_backend(cursor)
    # First move accepted immediately.
    cursor.move(10, 10)
    assert backend.position == (10, 10)
    # Second move within 10 ms is dropped.
    clock.advance(0.001)
    cursor.move(20, 20)
    assert backend.position == (10, 10)
    # After the min interval the move goes through.
    clock.advance(0.02)
    cursor.move(30, 30)
    assert backend.position == (30, 30)


def test_click_uses_button_enum() -> None:
    cursor = PynputCursor(rate_limit_hz=60, enabled=True)
    backend = _install_backend(cursor)
    cursor.click("left")
    cursor.click("right")
    assert backend.clicks == [("LEFT", 1), ("RIGHT", 1)]


def test_unknown_button_raises() -> None:
    cursor = PynputCursor(rate_limit_hz=60, enabled=True)
    _install_backend(cursor)
    with pytest.raises(IrisFlowError):
        cursor.click("teleport")


def test_move_swallows_backend_exception() -> None:
    cursor = PynputCursor(rate_limit_hz=60, enabled=True)

    class _AngryBackend:
        @property
        def position(self) -> tuple[int, int]:
            return (0, 0)

        @position.setter
        def position(self, value: tuple[int, int]) -> None:
            raise RuntimeError("no display")

        def click(self, button: object, count: int) -> None:
            raise RuntimeError("unused")

    cursor._backend = _AngryBackend()
    cursor._button_type = _FakeButton
    cursor.move(50, 50)  # must not raise


def test_click_swallows_backend_exception() -> None:
    cursor = PynputCursor(rate_limit_hz=60, enabled=True)

    class _AngryBackend(_FakeBackend):
        def click(self, button: object, count: int) -> None:
            raise RuntimeError("device busy")

    cursor._backend = _AngryBackend()
    cursor._button_type = _FakeButton
    cursor.click("left")  # must not raise


def test_missing_pynput_raises_iris_flow_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate pynput import failing.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pynput" or name.startswith("pynput."):
            raise ImportError("no pynput")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    cursor = PynputCursor(rate_limit_hz=60, enabled=True)
    with pytest.raises(IrisFlowError):
        cursor.move(10, 10)
