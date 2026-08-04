"""Assemble the per-run pipeline components from :class:`AppConfig`.

Sprint 7 only wires capture → detection → preprocess → inference. Later
sprints (calibration, mapping, filtering, control) extend
:class:`PipelineComponents` with their own fields.

The public factory :func:`build_pipeline` accepts optional overrides for
every component so tests can substitute fakes without monkey-patching
concrete constructors (the pattern the ``@pytest.mark.model`` suite
depends on).
"""

from __future__ import annotations

from dataclasses import dataclass

from irisflow.capture.webcam import WebcamSource
from irisflow.config.schema import AppConfig
from irisflow.core.clock import Clock, SystemClock
from irisflow.core.interfaces import FaceDetector, FrameSource, GazeEstimator, Preprocessor
from irisflow.detection.mediapipe_detector import MediaPipeFaceDetector, build_default_face_mesh
from irisflow.detection.tracking import TrackedFaceDetector
from irisflow.inference.registry import build_gaze_estimator
from irisflow.inference.warmup import warm_up_backend
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.state import PipelineStateMachine
from irisflow.preprocessing.builder import ModelInputBuilder
from irisflow.telemetry.metrics import MetricsRecorder

__all__ = ["PipelineComponents", "build_pipeline"]


@dataclass
class PipelineComponents:
    """Every collaborator the runner needs, wired for one session."""

    source: FrameSource
    detector: FaceDetector
    preprocessor: Preprocessor
    estimator: GazeEstimator
    bus: EventBus
    state: PipelineStateMachine
    metrics: MetricsRecorder
    clock: Clock


def build_pipeline(
    cfg: AppConfig,
    *,
    source: FrameSource | None = None,
    detector: FaceDetector | None = None,
    preprocessor: Preprocessor | None = None,
    estimator: GazeEstimator | None = None,
    bus: EventBus | None = None,
    clock: Clock | None = None,
    warmup: bool = True,
) -> PipelineComponents:
    """Construct a :class:`PipelineComponents` from ``cfg`` + overrides.

    Args:
        cfg: The full application config. Only the sections relevant to
            Sprint 7 (``camera``, ``detection``, ``model``) are consulted
            here; later sprints will read more.
        source, detector, preprocessor, estimator, bus, clock: Optional
            substitutes. Any component left as ``None`` is built from
            config using the default production factory.
        warmup: When true and the estimator was built here (not injected),
            call :func:`warm_up_backend` before returning.
    """
    real_clock: Clock = clock if clock is not None else SystemClock()
    real_bus = bus if bus is not None else EventBus()
    real_source = source if source is not None else _default_source(cfg, real_clock)
    real_detector = detector if detector is not None else _default_detector(cfg)
    real_preprocessor = (
        preprocessor if preprocessor is not None else _default_preprocessor(cfg)
    )
    if estimator is not None:
        real_estimator = estimator
    else:
        real_estimator = build_gaze_estimator(
            backend=cfg.model.backend,
            model_path=cfg.model.path,
            channel_order=cfg.model.channel_order,
            normalization=cfg.model.normalization,
        )
        if warmup:
            warm_up_backend(real_estimator, iterations=cfg.model.warmup_iterations)
    return PipelineComponents(
        source=real_source,
        detector=real_detector,
        preprocessor=real_preprocessor,
        estimator=real_estimator,
        bus=real_bus,
        state=PipelineStateMachine(real_bus, clock=real_clock),
        metrics=MetricsRecorder(_clock=real_clock),
        clock=real_clock,
    )


# ---------------------------------------------------------------------------
# Default constructors — kept private because the CLI/pipeline shouldn't
# know or care what class the concrete adapter is.
# ---------------------------------------------------------------------------
def _default_source(cfg: AppConfig, clock: Clock) -> FrameSource:
    return WebcamSource(
        device_id=cfg.camera.device_id,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        reconnect_backoff_ms=cfg.camera.reconnect_backoff_ms,
        clock=clock,
    )


def _default_detector(cfg: AppConfig) -> FaceDetector:
    mesh = build_default_face_mesh(
        model_path=cfg.detection.face_model_path,
        refine_landmarks=cfg.detection.refine_landmarks,
        min_detection_confidence=cfg.detection.min_detection_confidence,
        min_tracking_confidence=cfg.detection.min_tracking_confidence,
    )
    inner = MediaPipeFaceDetector(
        mesh, roi_margin=cfg.detection.roi_margin, square_rois=True
    )
    return TrackedFaceDetector(
        inner,
        smoothing_alpha=cfg.detection.smoothing_alpha,
        lost_hysteresis=cfg.detection.lost_hysteresis_frames,
    )


def _default_preprocessor(cfg: AppConfig) -> Preprocessor:
    return ModelInputBuilder(
        face_input_size=cfg.model.face_input_size,
        eye_input_size=cfg.model.eye_input_size,
        input_channel_order="BGR",  # OpenCV frames
        output_channel_order=cfg.model.channel_order,
        normalization=cfg.model.normalization,
    )
