"""Sprint 2 — clock injection: SystemClock delegates, FakeClock is deterministic."""

from __future__ import annotations

import pytest

from irisflow.core.clock import Clock, FakeClock, SystemClock


def test_system_clock_conforms_to_protocol() -> None:
    assert isinstance(SystemClock(), Clock)


def test_fake_clock_conforms_to_protocol() -> None:
    assert isinstance(FakeClock(), Clock)


def test_fake_clock_starts_at_configured_time() -> None:
    clock = FakeClock(start=42.0, wall=1_700_000_000.0)
    assert clock.monotonic() == 42.0
    assert clock.wall() == 1_700_000_000.0


def test_fake_clock_advance_moves_both_clocks() -> None:
    clock = FakeClock(start=10.0, wall=100.0)
    clock.advance(5.0)
    assert clock.monotonic() == 15.0
    assert clock.wall() == 105.0


def test_fake_clock_sleep_advances_instead_of_blocking() -> None:
    clock = FakeClock()
    clock.sleep(1.0)
    assert clock.monotonic() == 1.0


def test_fake_clock_rejects_negative_intervals() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="sleep seconds"):
        clock.sleep(-0.1)
    with pytest.raises(ValueError, match="advance seconds"):
        clock.advance(-1.0)


def test_fake_clock_sleep_zero_is_noop() -> None:
    clock = FakeClock(start=3.0)
    clock.sleep(0.0)
    assert clock.monotonic() == 3.0


def test_fake_clock_set_wall_leaves_monotonic_alone() -> None:
    clock = FakeClock(start=1.0, wall=100.0)
    clock.set_wall(500.0)
    assert clock.monotonic() == 1.0
    assert clock.wall() == 500.0


def test_system_clock_monotonic_is_non_decreasing() -> None:
    clock = SystemClock()
    a = clock.monotonic()
    b = clock.monotonic()
    assert b >= a


def test_system_clock_sleep_zero_does_not_raise() -> None:
    SystemClock().sleep(0.0)


def test_system_clock_wall_returns_positive_seconds() -> None:
    # We don't compare against `time.time()` (races the assertion); a plausible
    # ballpark is enough — this just needs to prove wall() is wired to a real
    # clock, not stuck at zero.
    assert SystemClock().wall() > 1_000_000_000.0


def test_system_clock_sleep_short_positive_interval_returns() -> None:
    # 1 ms is short enough not to slow the suite; the goal is just to prove
    # the `if seconds > 0` branch is exercised.
    SystemClock().sleep(0.001)
