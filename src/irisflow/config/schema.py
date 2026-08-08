"""Pydantic schema for every configuration section.

Fields mirror :file:`configs/default.yaml` one-to-one. Every field is validated
at load time; a missing or malformed value fails fast with a message that
names the offending path (see :func:`irisflow.config.loader.load_config`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NormalizationScheme = Literal["unit", "signed", "imagenet"]
ChannelOrder = Literal["RGB", "BGR"]
ModelBackend = Literal["keras", "onnx"]
CalibrationKind = Literal["affine", "polynomial", "ridge"]
FilterName = Literal["outlier", "ema", "one_euro", "kalman", "fixation"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _Strict(BaseModel):
    """Base class: forbid unknown fields so typos in YAML surface as errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class CameraConfig(_Strict):
    device_id: int = Field(0, ge=0, description="OpenCV VideoCapture index.")
    width: int = Field(1280, gt=0)
    height: int = Field(720, gt=0)
    fps: int = Field(30, gt=0, le=240)
    reconnect_backoff_ms: int = Field(500, ge=0)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
class DetectionConfig(_Strict):
    refine_landmarks: bool = True
    roi_margin: float = Field(0.20, ge=0.0, le=1.0)
    smoothing_alpha: float = Field(0.4, gt=0.0, le=1.0)
    lost_hysteresis_frames: int = Field(5, ge=0)
    face_model_path: Path = Path("models/face_landmarker.task")
    min_detection_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Model / Inference
# ---------------------------------------------------------------------------
class ModelConfig(_Strict):
    backend: ModelBackend = "keras"
    path: Path = Path("models/gaze_cnn_best.keras")
    warmup_iterations: int = Field(3, ge=0)
    channel_order: ChannelOrder = "RGB"
    normalization: NormalizationScheme = "unit"
    face_input_size: tuple[int, int] = (224, 224)
    eye_input_size: tuple[int, int] = (112, 112)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class CalibrationConfig(_Strict):
    points: Literal[5, 9, 13] = 9
    samples_per_point: int = Field(30, ge=1)
    stabilization_ms: int = Field(600, ge=0)
    model_kind: CalibrationKind = "polynomial"
    polynomial_degree: int = Field(2, ge=1, le=4)
    max_residual_px: float = Field(60.0, gt=0.0)
    profiles_dir: Path = Path("configs/profiles")


# ---------------------------------------------------------------------------
# Mapping (normalized gaze -> screen pixel)
# ---------------------------------------------------------------------------
class MappingConfig(_Strict):
    screen_id: int = Field(0, ge=0)
    clamp_margin_px: int = Field(8, ge=0)


# ---------------------------------------------------------------------------
# Filtering chain
# ---------------------------------------------------------------------------
class OutlierFilterConfig(_Strict):
    max_velocity_px_per_s: float = Field(6000.0, gt=0.0)


class EmaFilterConfig(_Strict):
    alpha: float = Field(0.4, gt=0.0, le=1.0)


class OneEuroConfig(_Strict):
    min_cutoff: float = Field(1.0, gt=0.0)
    beta: float = Field(0.007, ge=0.0)
    d_cutoff: float = Field(1.0, gt=0.0)


class FixationConfig(_Strict):
    velocity_threshold_px_per_s: float = Field(40.0, gt=0.0)


def _default_filter_chain() -> list[FilterName]:
    return ["outlier", "one_euro", "fixation"]


class FilterConfig(_Strict):
    chain: list[FilterName] = Field(default_factory=_default_filter_chain)
    outlier: OutlierFilterConfig = Field(default_factory=OutlierFilterConfig)
    ema: EmaFilterConfig = Field(default_factory=EmaFilterConfig)
    one_euro: OneEuroConfig = Field(default_factory=OneEuroConfig)
    fixation: FixationConfig = Field(default_factory=FixationConfig)


# ---------------------------------------------------------------------------
# Control (cursor + safety)
# ---------------------------------------------------------------------------
class DwellConfig(_Strict):
    radius_px: int = Field(40, gt=0)
    duration_ms: int = Field(800, gt=0)
    refractory_ms: int = Field(400, ge=0)


class SafetyConfig(_Strict):
    kill_switch: str = "ctrl+alt+esc"
    pause_on_face_lost_ms: int = Field(2000, ge=0)
    rest_zone_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    watchdog_timeout_ms: int = Field(500, ge=0)


class ControlConfig(_Strict):
    enabled: bool = False
    rate_limit_hz: int = Field(60, gt=0)
    dwell: DwellConfig = Field(default_factory=DwellConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
class TelemetryConfig(_Strict):
    metrics_enabled: bool = True
    record_sessions: bool = False
    recordings_dir: Path = Path("data/recordings")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class LoggingConfig(_Strict):
    level: LogLevel = "INFO"
    json_file: Path | None = Path("data/logs/irisflow.jsonl")
    human_console: bool = True


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class AppConfig(_Strict):
    """Top-level configuration — a single instance drives the whole system."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    filtering: FilterConfig = Field(default_factory=FilterConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


__all__ = [
    "AppConfig",
    "CalibrationConfig",
    "CameraConfig",
    "ChannelOrder",
    "ControlConfig",
    "DetectionConfig",
    "DwellConfig",
    "EmaFilterConfig",
    "FilterConfig",
    "FilterName",
    "FixationConfig",
    "LoggingConfig",
    "MappingConfig",
    "ModelBackend",
    "ModelConfig",
    "NormalizationScheme",
    "OneEuroConfig",
    "OutlierFilterConfig",
    "SafetyConfig",
    "TelemetryConfig",
]
