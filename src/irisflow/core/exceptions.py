"""Domain-wide exception hierarchy.

Everything the pipeline can raise inherits from :class:`IrisFlowError` so the
outer loop can distinguish "our problem" from unexpected third-party crashes.
Messages should always name what failed and, when relevant, how to fix it.
"""

from __future__ import annotations


class IrisFlowError(Exception):
    """Root of every error raised by the IrisFlow backend."""


class ConfigError(IrisFlowError):
    """Configuration is missing, malformed, or fails validation."""


class CaptureError(IrisFlowError):
    """Frame source is unavailable, disconnected, or returned invalid data."""


class DetectionError(IrisFlowError):
    """Face/eye detection failed in a way the pipeline cannot recover from."""


class InferenceError(IrisFlowError):
    """Gaze estimator backend failed to load or produced an invalid output."""


class CalibrationError(IrisFlowError):
    """Calibration session failed to converge or produced an unusable model."""


__all__ = [
    "CalibrationError",
    "CaptureError",
    "ConfigError",
    "DetectionError",
    "InferenceError",
    "IrisFlowError",
]
