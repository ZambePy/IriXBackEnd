"""Domain core: types, protocols, exceptions.

Only stdlib and numpy are allowed here — this package is imported by every
other layer, so it must stay dependency-free.
"""

from __future__ import annotations

from irisflow.core.exceptions import (
    CalibrationError,
    CaptureError,
    ConfigError,
    DetectionError,
    InferenceError,
    IrisFlowError,
)

__all__ = [
    "CalibrationError",
    "CaptureError",
    "ConfigError",
    "DetectionError",
    "InferenceError",
    "IrisFlowError",
]
