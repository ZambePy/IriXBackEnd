"""In-process deterministic :class:`FrameSource`.

Different from ``tests/fixtures/stubs.py::SyntheticFrameSource`` in two ways:

* It lives in ``src/`` so production code (``irisflow doctor``, demos, the
  future ``irisflow run --source synthetic``) can use it without the
  import-linter ``production code must not import test stubs`` contract
  slapping our wrist.
* It knows how to render distinguishable frame patterns (solid colour with
  frame id encoded in a channel; horizontal gradient; moving dot). Useful
  when eyeballing the pipeline without a webcam.

Frames are HxWx3 BGR ``uint8`` — same as OpenCV's :meth:`VideoCapture.read`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from irisflow.capture.base import BaseFrameSource
from irisflow.core.clock import Clock
from irisflow.core.types import Frame

Pattern = Literal["solid", "gradient", "moving_dot"]

__all__ = ["Pattern", "SyntheticFrameSource"]


class SyntheticFrameSource(BaseFrameSource):
    """Yields deterministic frames of a fixed size, indefinitely.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        pattern: Which visual pattern to render. All patterns encode the
            frame id somewhere so tests can distinguish successive frames.
        clock: Injected time source. Defaults to
            :class:`~irisflow.core.clock.SystemClock`; pass a
            :class:`~irisflow.core.clock.FakeClock` for deterministic tests.
    """

    __slots__ = ("_height", "_pattern", "_width")

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        pattern: Pattern = "solid",
        clock: Clock | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width/height must be positive, got {width}x{height}")
        super().__init__(clock=clock)
        self._width = width
        self._height = height
        self._pattern: Pattern = pattern

    def read(self) -> Frame | None:
        if not self._open:
            return None
        frame_id = self._next_frame_id()
        data = self._render(frame_id)
        return Frame(data=data, frame_id=frame_id, timestamp=self._clock.monotonic())

    def _render(self, frame_id: int) -> NDArray[np.uint8]:
        data = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        if self._pattern == "solid":
            data[..., 0] = frame_id % 255
        elif self._pattern == "gradient":
            column = np.linspace(0, 255, self._width, dtype=np.uint8)
            data[..., 1] = column
            data[..., 0] = frame_id % 255
        else:  # moving_dot
            data[..., 0] = 16
            cx = (frame_id * 3) % self._width
            cy = (frame_id * 2) % self._height
            r = 6
            y0, y1 = max(0, cy - r), min(self._height, cy + r)
            x0, x1 = max(0, cx - r), min(self._width, cx + r)
            data[y0:y1, x0:x1, 2] = 255
        return data
