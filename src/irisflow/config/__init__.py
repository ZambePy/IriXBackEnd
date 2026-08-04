"""Configuration loading and schema — see :mod:`irisflow.config.schema`."""

from __future__ import annotations

from irisflow.config.loader import load_config
from irisflow.config.schema import (
    AppConfig,
    CalibrationConfig,
    CameraConfig,
    ControlConfig,
    DetectionConfig,
    FilterConfig,
    LoggingConfig,
    MappingConfig,
    ModelConfig,
    TelemetryConfig,
)

__all__ = [
    "AppConfig",
    "CalibrationConfig",
    "CameraConfig",
    "ControlConfig",
    "DetectionConfig",
    "FilterConfig",
    "LoggingConfig",
    "MappingConfig",
    "ModelConfig",
    "TelemetryConfig",
    "load_config",
]
