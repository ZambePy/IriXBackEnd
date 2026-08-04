"""Domain core: types, protocols, geometry, clock, events, exceptions.

Only stdlib and numpy are allowed here — this package is imported by every
other layer, so it must stay dependency-free.
"""

from __future__ import annotations

from irisflow.core.clock import Clock, FakeClock, SystemClock
from irisflow.core.events import (
    CalibrationPhase,
    CalibrationProgress,
    Event,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    PipelineState,
    StateChanged,
)
from irisflow.core.exceptions import (
    CalibrationError,
    CaptureError,
    ConfigError,
    DetectionError,
    InferenceError,
    IrisFlowError,
)
from irisflow.core.geometry import (
    clamp,
    clip_to_frame,
    denormalize,
    expand,
    normalize,
    to_square,
)
from irisflow.core.interfaces import (
    CalibrationModel,
    CursorController,
    FaceDetector,
    Filter,
    FrameSource,
    GazeEstimator,
    Preprocessor,
    ScreenMapper,
)
from irisflow.core.types import (
    BoundingBox,
    CalibratedGaze,
    FaceDetection,
    Frame,
    ModelInput,
    ModelMetadata,
    PipelineTick,
    RawGaze,
    ScreenPoint,
    SmoothedPoint,
)

__all__ = [
    "BoundingBox",
    "CalibratedGaze",
    "CalibrationError",
    "CalibrationModel",
    "CalibrationPhase",
    "CalibrationProgress",
    "CaptureError",
    "Clock",
    "ConfigError",
    "CursorController",
    "DetectionError",
    "Event",
    "FaceAcquired",
    "FaceDetection",
    "FaceDetector",
    "FaceLost",
    "FakeClock",
    "Filter",
    "Frame",
    "FrameSource",
    "GazeEstimator",
    "GazeUpdated",
    "InferenceError",
    "IrisFlowError",
    "ModelInput",
    "ModelMetadata",
    "PipelineState",
    "PipelineTick",
    "Preprocessor",
    "RawGaze",
    "ScreenMapper",
    "ScreenPoint",
    "SmoothedPoint",
    "StateChanged",
    "SystemClock",
    "clamp",
    "clip_to_frame",
    "denormalize",
    "expand",
    "normalize",
    "to_square",
]
