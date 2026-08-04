"""Sprint 3 — deterministic synthetic frame source."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.core.clock import FakeClock


def test_synthetic_source_returns_none_when_closed() -> None:
    source = SyntheticFrameSource()
    assert source.read() is None


def test_synthetic_source_yields_frames_with_configured_shape() -> None:
    source = SyntheticFrameSource(width=320, height=240)
    with source:
        frame = source.read()
    assert frame is not None
    assert frame.width == 320
    assert frame.height == 240
    assert frame.data.dtype == np.uint8


def test_synthetic_source_uses_injected_clock_for_timestamps() -> None:
    clock = FakeClock()
    source = SyntheticFrameSource(clock=clock)
    with source:
        first = source.read()
        clock.advance(0.1)
        second = source.read()
    assert first is not None
    assert second is not None
    assert first.timestamp == pytest.approx(0.0)
    assert second.timestamp == pytest.approx(0.1)


def test_synthetic_source_produces_monotonic_frame_ids() -> None:
    source = SyntheticFrameSource()
    with source:
        ids = [source.read().frame_id for _ in range(5)]  # type: ignore[union-attr]
    assert ids == [0, 1, 2, 3, 4]


def test_gradient_pattern_encodes_horizontal_ramp() -> None:
    source = SyntheticFrameSource(width=16, height=4, pattern="gradient")
    with source:
        frame = source.read()
    assert frame is not None
    row = frame.data[0, :, 1]
    assert row[0] < row[-1]


def test_moving_dot_pattern_shifts_across_frames() -> None:
    source = SyntheticFrameSource(width=64, height=64, pattern="moving_dot")
    with source:
        a = source.read()
        b = source.read()
    assert a is not None
    assert b is not None
    # Different frames must not be pixel-equal — the dot has moved.
    assert not np.array_equal(a.data, b.data)


def test_invalid_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="positive"):
        SyntheticFrameSource(width=0, height=100)
    with pytest.raises(ValueError, match="positive"):
        SyntheticFrameSource(width=100, height=-1)


def test_close_is_idempotent() -> None:
    source = SyntheticFrameSource()
    source.open()
    source.close()
    source.close()
    assert not source.is_open


def test_open_is_idempotent() -> None:
    source = SyntheticFrameSource()
    source.open()
    source.open()
    assert source.is_open
    source.close()
