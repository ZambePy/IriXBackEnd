"""In-memory fake of :class:`cv2.VideoCapture` for capture-layer tests.

Kept in ``tests/fixtures/`` so the ``production code must not import test
stubs`` import-linter contract stays green — production code always talks
to the real ``cv2.VideoCapture`` (or a factory injected by the CLI, never
by ``src/``).

The fake supports the three failure modes the webcam source must handle:

* Never opens (``open_ok=False``): simulates a device that enumerates but
  is unusable — the source should keep retrying with backoff.
* Opens then read fails after N frames (``fail_after``): simulates a
  hot-unplug — the source should release, count a reconnect, and retry.
* Opens and delivers frames forever (default): happy path.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray


class FakeVideoCapture:
    """Minimal drop-in for :class:`cv2.VideoCapture` used by capture tests."""

    def __init__(
        self,
        device_id: int,
        *,
        open_ok: bool = True,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        fail_after: int | None = None,
        backend_name: str = "FAKE",
    ) -> None:
        self.device_id = device_id
        self._open = open_ok
        self._width = width
        self._height = height
        self._fps = fps
        self._fail_after = fail_after
        self._reads = 0
        self._released = False
        self._backend = backend_name
        self.props: dict[int, float] = {
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_FPS: float(fps),
        }

    # ---- cv2.VideoCapture API surface used by src/irisflow/capture/
    def isOpened(self) -> bool:  # noqa: N802 — mimic cv2 API
        return self._open and not self._released

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if not self.isOpened():
            return False, None
        if self._fail_after is not None and self._reads >= self._fail_after:
            self._open = False
            return False, None
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[..., 0] = self._reads % 255
        self._reads += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        return float(self.props.get(prop_id, 0.0))

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = float(value)
        return True

    def release(self) -> None:
        self._released = True
        self._open = False

    def getBackendName(self) -> str:  # noqa: N802 — mimic cv2 API
        return self._backend


def make_factory(
    *,
    open_ok: bool = True,
    fail_after: int | None = None,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> Callable[[int], FakeVideoCapture]:
    """Build a ``capture_factory`` closure with the given behaviour."""

    def _factory(device_id: int) -> FakeVideoCapture:
        return FakeVideoCapture(
            device_id,
            open_ok=open_ok,
            fail_after=fail_after,
            width=width,
            height=height,
            fps=fps,
        )

    return _factory


def make_flaky_factory(
    *, reconnect_after: int
) -> tuple[Callable[[int], FakeVideoCapture], threading.Event]:
    """Factory whose first capture disconnects; second stays healthy.

    The returned event is set once the second (healthy) capture has been
    handed out — tests can wait on it to know the reconnect happened
    without polling ``source.reconnect_count`` in a busy loop.
    """
    reconnected = threading.Event()
    calls: list[int] = []

    def _factory(device_id: int) -> FakeVideoCapture:
        calls.append(device_id)
        if len(calls) == 1:
            return FakeVideoCapture(device_id, fail_after=reconnect_after)
        reconnected.set()
        return FakeVideoCapture(device_id)

    return _factory, reconnected


__all__ = [
    "FakeVideoCapture",
    "make_factory",
    "make_flaky_factory",
]
