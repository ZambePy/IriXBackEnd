"""Sprint 3 — device enumeration and empirical FPS measurement."""

from __future__ import annotations

import pytest

from irisflow.capture.diagnostics import (
    enumerate_cameras,
    measure_capture,
)
from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.core.clock import FakeClock
from tests.fixtures.fake_capture import FakeVideoCapture


def _mixed_factory(device_id: int) -> FakeVideoCapture:
    # id 0 opens, id 1 does not, id 2 opens with different resolution.
    if device_id == 0:
        return FakeVideoCapture(device_id, width=640, height=480, fps=30.0)
    if device_id == 1:
        return FakeVideoCapture(device_id, open_ok=False)
    return FakeVideoCapture(device_id, width=1280, height=720, fps=60.0)


def test_enumerate_cameras_reports_availability_per_index() -> None:
    cams = enumerate_cameras(max_index=3, capture_factory=_mixed_factory)
    assert [c.device_id for c in cams] == [0, 1, 2]
    assert [c.available for c in cams] == [True, False, True]
    assert cams[0].width == 640
    assert cams[2].fps_reported == 60.0


def test_enumerate_cameras_survives_factory_exceptions() -> None:
    def boom_factory(device_id: int) -> FakeVideoCapture:
        raise OSError(f"denied {device_id}")

    cams = enumerate_cameras(max_index=2, capture_factory=boom_factory)
    assert len(cams) == 2
    assert all(not c.available for c in cams)


def test_enumerate_cameras_rejects_bad_max_index() -> None:
    with pytest.raises(ValueError, match="max_index"):
        enumerate_cameras(max_index=0)


def test_measure_capture_counts_frames_over_duration() -> None:
    # Use the real clock here — measure_capture is designed to measure
    # wall time, and a FakeClock only advances via sleep(), which the
    # happy path (read returns a frame) never triggers.
    source = SyntheticFrameSource()
    source.open()
    try:
        result = measure_capture(source, duration_s=0.2, poll_interval_s=0.0)
    finally:
        source.close()
    assert result.frames_captured > 0
    assert result.duration_s >= 0.2
    assert result.fps_measured > 0


def test_measure_capture_reports_zero_when_source_never_yields() -> None:
    class _AlwaysNone:
        def open(self) -> None:  # pragma: no cover - unused
            pass

        def close(self) -> None:  # pragma: no cover - unused
            pass

        def read(self):  # type: ignore[no-untyped-def]
            return None

        @property
        def is_open(self) -> bool:  # pragma: no cover - unused
            return True

    clock = FakeClock()
    result = measure_capture(
        _AlwaysNone(),  # type: ignore[arg-type]
        duration_s=0.5,
        clock=clock,
        poll_interval_s=0.05,
    )
    assert result.frames_captured == 0
    assert result.fps_measured == 0.0
    assert result.mean_read_latency_ms == 0.0


def test_measure_capture_validates_duration() -> None:
    source = SyntheticFrameSource()
    source.open()
    try:
        with pytest.raises(ValueError, match="duration_s"):
            measure_capture(source, duration_s=0)
    finally:
        source.close()
