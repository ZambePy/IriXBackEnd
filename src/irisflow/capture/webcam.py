"""Webcam :class:`FrameSource` with a dedicated capture thread.

Design notes (SPRINTS.md §2.2 and §4):

* A dedicated background thread pulls frames from the OS at whatever rate
  the driver allows. Consumers (the pipeline) call :meth:`read` from any
  thread and get the **most recent** frame — or ``None`` if none has arrived
  yet.
* The internal buffer is a single-slot queue with **drop-oldest** semantics.
  If the pipeline lags behind the camera we want the newest frame, never a
  backlog: latency matters more than completeness.
* Disconnect is a normal condition, not a fatal error. When
  :meth:`cv2.VideoCapture.read` starts failing, the thread releases the
  device, waits ``reconnect_backoff_ms`` and tries again — the process
  keeps running so a user with ALS is never trapped by a loose USB cable.
* ``open`` starts the thread and returns immediately; ``read`` returns
  ``None`` until the first frame is buffered.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

import cv2

from irisflow.capture.base import BaseFrameSource
from irisflow.core.clock import Clock
from irisflow.core.types import Frame
from irisflow.logging import get_logger

__all__ = ["CaptureFactory", "WebcamSource"]


CaptureFactory = Callable[[int], Any]
"""``int → cv2.VideoCapture-like``. Tests inject a fake to avoid touching hardware."""


_DEFAULT_JOIN_TIMEOUT_S = 2.0
_STOP_POLL_S = 0.05


class WebcamSource(BaseFrameSource):
    """OpenCV-backed webcam source with background capture and reconnection.

    Args:
        device_id: OpenCV :class:`~cv2.VideoCapture` index (0 = default cam).
        width, height, fps: Requested capture properties. The driver may
            silently ignore them — :meth:`~cv2.VideoCapture.get` after open
            returns what was actually applied.
        reconnect_backoff_ms: Wait between reconnect attempts after a read
            failure. Zero means "retry as fast as possible" (not recommended
            outside tests — will busy-loop on a bad camera).
        capture_factory: Override the ``cv2.VideoCapture`` constructor;
            tests use this to inject a deterministic fake capture object.
        clock: Injected time source; see :class:`BaseFrameSource`.
    """

    __slots__ = (
        "_backoff_s",
        "_cap",
        "_cap_lock",
        "_capture_factory",
        "_device_id",
        "_fps",
        "_height",
        "_last_error",
        "_log",
        "_queue",
        "_reconnect_count",
        "_stop",
        "_thread",
        "_width",
    )

    def __init__(
        self,
        *,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        reconnect_backoff_ms: int = 500,
        capture_factory: CaptureFactory | None = None,
        clock: Clock | None = None,
    ) -> None:
        if device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {device_id}")
        if width <= 0 or height <= 0:
            raise ValueError(f"width/height must be positive, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if reconnect_backoff_ms < 0:
            raise ValueError(f"reconnect_backoff_ms must be >= 0, got {reconnect_backoff_ms}")
        super().__init__(clock=clock)
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._backoff_s = reconnect_backoff_ms / 1000.0
        self._capture_factory: CaptureFactory = (
            capture_factory if capture_factory is not None else cv2.VideoCapture
        )
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._cap_lock = threading.Lock()
        self._cap: Any | None = None
        self._thread: threading.Thread | None = None
        self._reconnect_count = 0
        self._last_error: str | None = None
        self._log = get_logger("irisflow.capture.webcam")

    # ---------------------------------------------------------------- lifecycle
    def _do_open(self) -> None:
        self._stop.clear()
        self._reconnect_count = 0
        self._last_error = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"irisflow-capture-{self._device_id}",
            daemon=True,
        )
        self._thread.start()

    def _do_close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=_DEFAULT_JOIN_TIMEOUT_S)
        with self._cap_lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                self._log.warning("capture.release_failed", device_id=self._device_id)
        self._drain_queue()

    # ---------------------------------------------------------------- consumer
    def read(self) -> Frame | None:
        """Return the most recent buffered frame, or ``None`` if none yet."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    # ---------------------------------------------------------------- stats
    @property
    def reconnect_count(self) -> int:
        """How many times the capture thread has reopened the device."""
        return self._reconnect_count

    @property
    def last_error(self) -> str | None:
        """Human-readable last error from the capture thread, or ``None``."""
        return self._last_error

    # ---------------------------------------------------------------- thread body
    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._ensure_connected():
                self._sleep_backoff()
                continue
            frame = self._grab_one()
            if frame is None:
                # Read failed — device likely disconnected. Reset and retry.
                self._release_locked()
                self._sleep_backoff()
                continue
            self._publish(frame)

    def _ensure_connected(self) -> bool:
        with self._cap_lock:
            if self._cap is not None and self._is_opened(self._cap):
                return True
            self._cap = None
        try:
            cap = self._capture_factory(self._device_id)
        except Exception as exc:
            self._last_error = f"factory raised: {exc}"
            self._log.warning(
                "capture.open_failed", device_id=self._device_id, error=str(exc)
            )
            return False
        if not self._is_opened(cap):
            with contextlib.suppress(Exception):
                cap.release()
            self._last_error = f"device {self._device_id} not available"
            self._log.warning("capture.open_failed", device_id=self._device_id)
            return False
        # Best-effort property configuration. Drivers often ignore these
        # silently — measurement in `diagnostics.measure_capture` is the
        # source of truth for what actually came out.
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, self._width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self._height),
            (cv2.CAP_PROP_FPS, self._fps),
        ):
            with contextlib.suppress(Exception):
                cap.set(prop, value)
        with self._cap_lock:
            self._cap = cap
        if self._reconnect_count > 0:
            self._log.info(
                "capture.reconnected",
                device_id=self._device_id,
                attempt=self._reconnect_count,
            )
        return True

    def _grab_one(self) -> Frame | None:
        with self._cap_lock:
            cap = self._cap
        if cap is None:
            return None
        try:
            ok, data = cap.read()
        except Exception as exc:
            self._last_error = f"read raised: {exc}"
            self._log.warning(
                "capture.read_raised", device_id=self._device_id, error=str(exc)
            )
            return None
        if not ok or data is None:
            self._last_error = "read returned no data"
            return None
        return Frame(
            data=data,
            frame_id=self._next_frame_id(),
            timestamp=self._clock.monotonic(),
        )

    def _publish(self, frame: Frame) -> None:
        # Drop-oldest: if the slot is full, discard the stale frame and put the
        # newest one in its place. Latency > completeness (SPRINTS.md §2.2).
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        with contextlib.suppress(queue.Empty):
            self._queue.get_nowait()
        # Somebody may have consumed just now — swallow and let the next tick win.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(frame)

    def _release_locked(self) -> None:
        self._reconnect_count += 1
        with self._cap_lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()
        self._log.warning(
            "capture.disconnected",
            device_id=self._device_id,
            error=self._last_error,
        )

    def _sleep_backoff(self) -> None:
        # We honour the stop event during backoff so ``close()`` doesn't have
        # to wait a full backoff cycle before the thread exits.
        remaining = self._backoff_s
        while remaining > 0 and not self._stop.is_set():
            chunk = min(_STOP_POLL_S, remaining)
            self._clock.sleep(chunk)
            remaining -= chunk

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _is_opened(cap: Any) -> bool:
        try:
            return bool(cap.isOpened())
        except Exception:
            return False
