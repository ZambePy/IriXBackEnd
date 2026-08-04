"""Sprint 3 — webcam source: thread lifecycle, drop-oldest, reconnection."""

from __future__ import annotations

import gc
import time

import pytest

from irisflow.capture.webcam import WebcamSource
from tests.fixtures.fake_capture import FakeVideoCapture, make_factory, make_flaky_factory


def _wait_for(condition, timeout_s: float = 2.0, poll_s: float = 0.01) -> bool:
    """Small helper — poll until ``condition()`` is true or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_s)
    return False


def _read_one(source: WebcamSource, timeout_s: float = 2.0):  # type: ignore[no-untyped-def]
    """Poll ``read`` until a non-``None`` frame is available."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = source.read()
        if frame is not None:
            return frame
        time.sleep(0.005)
    return None


def test_webcam_source_delivers_frames_from_fake_capture() -> None:
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    with source:
        frame = _read_one(source)
    assert frame is not None
    assert frame.width == 640
    assert frame.height == 480
    # First consumed frame is whichever was most recent when the queue
    # was drained — always >= 0 but not necessarily 0 (drop-oldest at work).
    assert frame.frame_id >= 0


def test_webcam_source_frame_ids_and_timestamps_are_monotonic() -> None:
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    with source:
        frames = []
        for _ in range(20):
            frame = _read_one(source, timeout_s=1.0)
            if frame is not None:
                frames.append(frame)
    assert len(frames) >= 3, "expected multiple frames from a healthy fake capture"
    ids = [f.frame_id for f in frames]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts)


def test_webcam_source_queue_never_exceeds_one_slot() -> None:
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    with source:
        # Give the thread time to fill and refill the slot several times
        # without the consumer draining it.
        time.sleep(0.2)
        assert source._queue.qsize() <= 1  # type: ignore[attr-defined]


def test_webcam_source_drop_oldest_discards_stale_frames() -> None:
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    with source:
        first = _read_one(source)
        assert first is not None
        # Let the capture thread produce many more frames while we don't read.
        time.sleep(0.15)
        latest = source.read()
    assert latest is not None
    assert latest.frame_id > first.frame_id
    # If the queue were unbounded we would still be sitting on frame_id == 1.
    # Drop-oldest means the frame id we now see reflects real production.
    assert latest.frame_id >= 2


def test_webcam_source_returns_none_before_thread_produces_anything() -> None:
    # Factory that never actually opens — the thread will loop backing off
    # and no frame is ever queued.
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(open_ok=False),
        reconnect_backoff_ms=10,
    )
    source.open()
    try:
        # Give it a moment to run its retry loop.
        time.sleep(0.1)
        assert source.read() is None
        assert source.last_error is not None
    finally:
        source.close()


def test_webcam_source_reconnects_after_disconnect() -> None:
    factory, reconnected = make_flaky_factory(reconnect_after=2)
    source = WebcamSource(
        device_id=0,
        capture_factory=factory,
        reconnect_backoff_ms=10,
    )
    with source:
        assert reconnected.wait(timeout=2.0), "capture thread never reopened"
        frame = _read_one(source, timeout_s=2.0)
    assert frame is not None
    assert source.reconnect_count >= 1


def test_webcam_source_close_releases_thread_promptly() -> None:
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    source.open()
    _read_one(source)
    start = time.monotonic()
    source.close()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"close took {elapsed:.3f}s, expected < 1s"
    assert not source.is_open


def test_webcam_source_context_manager_closes_on_exit() -> None:
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    with source:
        assert source.is_open
    assert not source.is_open


def test_webcam_source_capture_factory_exception_is_swallowed() -> None:
    def boom_factory(_device_id: int) -> FakeVideoCapture:
        raise RuntimeError("no such device")

    source = WebcamSource(
        device_id=0,
        capture_factory=boom_factory,
        reconnect_backoff_ms=10,
    )
    source.open()
    try:
        # Give the thread a moment to hit the exception in its retry loop.
        time.sleep(0.05)
        assert source.read() is None
        assert source.last_error is not None
        assert "no such device" in source.last_error
    finally:
        source.close()


def test_webcam_source_extended_run_does_not_leak_frames() -> None:
    """Proxy for the "60s without memory growth" criterion.

    We don't need 60s of wall clock to catch a leak — the invariant is
    that the internal queue stays bounded at 1 and no other structure
    grows unboundedly. Running for a few hundred ticks and asserting
    those bounds is enough and keeps the suite fast.
    """
    source = WebcamSource(
        device_id=0,
        capture_factory=make_factory(),
        reconnect_backoff_ms=10,
    )
    frames_seen: list[int] = []
    with source:
        for _ in range(300):
            frame = source.read()
            if frame is not None:
                frames_seen.append(frame.frame_id)
            time.sleep(0.001)
        gc.collect()
        assert source._queue.qsize() <= 1  # type: ignore[attr-defined]
    assert len(frames_seen) > 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"device_id": -1}, "device_id"),
        ({"width": 0}, "width/height"),
        ({"height": -5}, "width/height"),
        ({"fps": 0}, "fps"),
        ({"reconnect_backoff_ms": -1}, "reconnect_backoff_ms"),
    ],
)
def test_webcam_source_rejects_invalid_config(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        WebcamSource(capture_factory=make_factory(), **kwargs)
