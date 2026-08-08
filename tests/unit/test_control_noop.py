"""Sprint 10 — the production NoOpCursor (distinct from the test fixture)."""

from __future__ import annotations

from irisflow.control.noop import NoOpCursor
from irisflow.core.interfaces import CursorController


def test_noop_cursor_is_a_cursor_controller() -> None:
    cursor: CursorController = NoOpCursor()
    assert isinstance(cursor, CursorController)


def test_noop_cursor_starts_enabled_by_default() -> None:
    assert NoOpCursor().is_enabled is True


def test_noop_cursor_move_and_click_are_no_ops() -> None:
    cursor = NoOpCursor()
    cursor.move(100, 100)  # must not raise
    cursor.click()
    cursor.click("right")


def test_noop_cursor_disable_toggles_is_enabled() -> None:
    cursor = NoOpCursor()
    cursor.disable()
    assert not cursor.is_enabled
    cursor.enable()
    assert cursor.is_enabled


def test_noop_cursor_disabled_at_construction() -> None:
    cursor = NoOpCursor(enabled=False)
    assert not cursor.is_enabled
    cursor.enable()
    assert cursor.is_enabled
