"""Enumerate cameras and measure the real capture characteristics.

Two things the OS-declared numbers cannot tell you:

* Which of the ``/dev/video*`` (or DirectShow / AVFoundation) indices
  actually produce frames — a lot of laptops enumerate IR sensors, virtual
  cameras and unusable placeholders alongside the real webcam.
* What frame rate the camera really delivers. The driver-reported FPS is
  a hint at best; measured FPS after a warm-up is what the pipeline sees.

Both facts feed into ``irisflow doctor`` (see :mod:`irisflow.cli.commands.doctor`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2

from irisflow.core.clock import Clock, SystemClock
from irisflow.core.interfaces import FrameSource

__all__ = [
    "CameraInfo",
    "CaptureMeasurement",
    "enumerate_cameras",
    "measure_capture",
]


CaptureFactory = Callable[[int], Any]
"""``int → cv2.VideoCapture-like``; injectable for tests without hardware."""


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """One row of the capture-device enumeration.

    ``available`` is what actually matters: a device may enumerate but
    return ``False`` from :meth:`cv2.VideoCapture.isOpened` on some systems.
    """

    device_id: int
    available: bool
    width: int
    height: int
    fps_reported: float
    backend: str


@dataclass(frozen=True, slots=True)
class CaptureMeasurement:
    """Empirical measurement of a running :class:`FrameSource`.

    ``mean_read_latency_ms`` is measured wall-time between successive
    non-``None`` reads; ``fps_measured`` is derived from ``frames_captured``
    over ``duration_s`` (not from any driver-reported value).
    """

    frames_captured: int
    duration_s: float
    fps_measured: float
    mean_read_latency_ms: float


def enumerate_cameras(
    *,
    max_index: int = 5,
    capture_factory: CaptureFactory | None = None,
) -> list[CameraInfo]:
    """Probe device indices ``[0, max_index)`` and return what we found.

    Args:
        max_index: Stop probing after this many indices. Increase carefully:
            each probe opens the device, which can be slow (~200 ms per index
            on Windows DirectShow).
        capture_factory: Override the ``cv2.VideoCapture`` constructor;
            tests inject a fake to avoid touching real hardware.
    """
    if max_index <= 0:
        raise ValueError(f"max_index must be > 0, got {max_index}")
    factory: CaptureFactory = (
        capture_factory if capture_factory is not None else cv2.VideoCapture
    )
    results: list[CameraInfo] = []
    for device_id in range(max_index):
        try:
            cap = factory(device_id)
        except Exception:
            results.append(
                CameraInfo(
                    device_id=device_id,
                    available=False,
                    width=0,
                    height=0,
                    fps_reported=0.0,
                    backend="unavailable",
                )
            )
            continue
        available = _safe_is_opened(cap)
        if not available:
            _safe_release(cap)
            results.append(
                CameraInfo(
                    device_id=device_id,
                    available=False,
                    width=0,
                    height=0,
                    fps_reported=0.0,
                    backend="unavailable",
                )
            )
            continue
        width = int(_safe_get(cap, cv2.CAP_PROP_FRAME_WIDTH))
        height = int(_safe_get(cap, cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(_safe_get(cap, cv2.CAP_PROP_FPS))
        backend = _safe_backend_name(cap)
        _safe_release(cap)
        results.append(
            CameraInfo(
                device_id=device_id,
                available=True,
                width=width,
                height=height,
                fps_reported=fps,
                backend=backend,
            )
        )
    return results


def measure_capture(
    source: FrameSource,
    *,
    duration_s: float = 2.0,
    clock: Clock | None = None,
    poll_interval_s: float = 0.005,
) -> CaptureMeasurement:
    """Run ``source`` for ``duration_s`` and report what came out.

    Assumes the source is already ``open``. Does not close it — the caller
    owns the lifecycle (this is what lets ``irisflow doctor`` measure a
    source that will then be handed off to the pipeline).

    Args:
        source: Any object satisfying the :class:`FrameSource` protocol.
        duration_s: How long to sample, in seconds. Longer is more accurate.
        clock: Time source; defaults to :class:`SystemClock`. Tests pass a
            :class:`~irisflow.core.clock.FakeClock` for determinism.
        poll_interval_s: Sleep between polls when the source returns
            ``None`` — keeps the loop from spinning on an idle source.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if poll_interval_s < 0:
        raise ValueError(f"poll_interval_s must be >= 0, got {poll_interval_s}")
    real_clock: Clock = clock if clock is not None else SystemClock()

    start = real_clock.monotonic()
    deadline = start + duration_s
    frames = 0
    prev_ts: float | None = None
    latencies: list[float] = []

    while True:
        now = real_clock.monotonic()
        if now >= deadline:
            break
        frame = source.read()
        if frame is None:
            if poll_interval_s > 0:
                real_clock.sleep(poll_interval_s)
            continue
        frames += 1
        ts = real_clock.monotonic()
        if prev_ts is not None:
            latencies.append((ts - prev_ts) * 1000.0)
        prev_ts = ts

    duration = max(real_clock.monotonic() - start, 1e-9)
    fps_measured = frames / duration if duration > 0 else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return CaptureMeasurement(
        frames_captured=frames,
        duration_s=duration,
        fps_measured=fps_measured,
        mean_read_latency_ms=mean_latency,
    )


# ---------------------------------------------------------------------------
# Small defensive helpers — cv2 tends to raise or return unexpected types on
# unusual hardware; we never let a probe crash the caller.
# ---------------------------------------------------------------------------
def _safe_is_opened(cap: Any) -> bool:
    try:
        return bool(cap.isOpened())
    except Exception:
        return False


def _safe_release(cap: Any) -> None:
    with contextlib.suppress(Exception):
        cap.release()


def _safe_get(cap: Any, prop: int) -> float:
    try:
        return float(cap.get(prop))
    except Exception:
        return 0.0


def _safe_backend_name(cap: Any) -> str:
    try:
        return str(cap.getBackendName())
    except Exception:
        return "unknown"
