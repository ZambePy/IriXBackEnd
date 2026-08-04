"""File-backed :class:`FrameSource` for tests, replays and offline benchmarks.

A video file is deterministic — every run produces the same frames in the
same order — which makes it the natural fixture for integration and E2E
tests that need "real" pixels without a webcam. It is also what the future
``irisflow replay <session>`` (Sprint 11) will use as its input.

Timestamps are read from :func:`Clock.monotonic` at read time (not from the
file's PTS) because downstream latency measurements only make sense when
timestamps come from the same monotonic origin as the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from irisflow.capture.base import BaseFrameSource
from irisflow.core.clock import Clock
from irisflow.core.exceptions import CaptureError
from irisflow.core.types import Frame

__all__ = ["VideoFileSource"]


class VideoFileSource(BaseFrameSource):
    """Sequentially reads frames from a video file.

    Args:
        path: File path. Must exist and be openable by OpenCV.
        loop: When true, seek back to frame 0 on end-of-file so ``read``
            never returns ``None`` for exhaustion. Useful for long-running
            integration tests over short fixtures.
        clock: Injected time source; see :class:`BaseFrameSource`.
    """

    __slots__ = ("_cap", "_loop", "_path")

    def __init__(
        self,
        path: Path | str,
        *,
        loop: bool = False,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._path = Path(path)
        self._loop = loop
        self._cap: cv2.VideoCapture | None = None

    def _do_open(self) -> None:
        if not self._path.exists():
            raise CaptureError(f"Video file not found: {self._path}")
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"OpenCV could not open video file: {self._path}")
        self._cap = cap

    def _do_close(self) -> None:
        cap = self._cap
        self._cap = None
        if cap is not None:
            cap.release()

    def read(self) -> Frame | None:
        cap = self._cap
        if cap is None:
            return None
        ok, data = cap.read()
        if not ok or data is None:
            if not self._loop:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, data = cap.read()
            if not ok or data is None:
                return None
        return Frame(
            data=np.ascontiguousarray(data, dtype=np.uint8),
            frame_id=self._next_frame_id(),
            timestamp=self._clock.monotonic(),
        )
