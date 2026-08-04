"""Capture layer — every concrete :class:`~irisflow.core.interfaces.FrameSource`.

Public surface:

* :class:`WebcamSource` — real OpenCV webcam with a background thread and
  drop-oldest buffering (production).
* :class:`VideoFileSource` — deterministic playback from disk (tests, replay).
* :class:`SyntheticFrameSource` — in-process procedurally generated frames
  (headless demos, ``irisflow doctor`` on a machine without a camera).
* :mod:`diagnostics` — enumerate devices and measure real FPS / latency.
"""

from irisflow.capture.diagnostics import (
    CameraInfo,
    CaptureMeasurement,
    enumerate_cameras,
    measure_capture,
)
from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.capture.video_file import VideoFileSource
from irisflow.capture.webcam import WebcamSource

__all__ = [
    "CameraInfo",
    "CaptureMeasurement",
    "SyntheticFrameSource",
    "VideoFileSource",
    "WebcamSource",
    "enumerate_cameras",
    "measure_capture",
]
