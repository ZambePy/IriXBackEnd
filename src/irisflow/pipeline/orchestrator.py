"""Assemble the per-run pipeline components from :class:`AppConfig`.

Sprint 7 wired capture → detection → preprocess → inference. Sprint 10
adds the missing downstream stages: calibration → mapping → filtering →
control. Every stage stays optional so tests can still bring up a
minimal loop with :class:`~tests.fixtures.stubs.StubGazeEstimator` and
skip the rest.

The public factory :func:`build_pipeline` accepts optional overrides for
every component so tests can substitute fakes without monkey-patching
concrete constructors (the pattern the ``@pytest.mark.model`` suite
depends on).
"""

from __future__ import annotations

from dataclasses import dataclass

from irisflow.calibration.models import build_calibration_model
from irisflow.calibration.store import CalibrationStore
from irisflow.capture.webcam import WebcamSource
from irisflow.config.schema import AppConfig
from irisflow.control.noop import NoOpCursor
from irisflow.core.clock import Clock, SystemClock
from irisflow.core.exceptions import CalibrationError
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
from irisflow.detection.mediapipe_detector import MediaPipeFaceDetector, build_default_face_mesh
from irisflow.detection.tracking import TrackedFaceDetector
from irisflow.filtering.chain import (
    ChainConfig,
    EmaParams,
    FilterName,
    FixationParams,
    OneEuroParams,
    OutlierParams,
    build_filter_chain,
)
from irisflow.inference.registry import build_gaze_estimator
from irisflow.inference.warmup import warm_up_backend
from irisflow.mapping.screen import ScreenMapper as ScreenPixelMapper
from irisflow.mapping.screen_info import ScreenInfo
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.state import PipelineStateMachine
from irisflow.preprocessing.builder import ModelInputBuilder
from irisflow.telemetry.metrics import MetricsRecorder

__all__ = ["PipelineComponents", "build_pipeline", "filter_config_from_app"]


@dataclass
class PipelineComponents:
    """Every collaborator the runner needs, wired for one session.

    ``calibration_model``, ``mapper``, ``filter_chain`` and ``cursor``
    are populated by :func:`build_pipeline`. Every one of them may be
    swapped for a stub in tests; the runner only uses them when present.
    """

    source: FrameSource
    detector: FaceDetector
    preprocessor: Preprocessor
    estimator: GazeEstimator
    bus: EventBus
    state: PipelineStateMachine
    metrics: MetricsRecorder
    clock: Clock
    calibration_model: CalibrationModel | None = None
    mapper: ScreenMapper | None = None
    filter_chain: Filter | None = None
    cursor: CursorController | None = None


def build_pipeline(
    cfg: AppConfig,
    *,
    source: FrameSource | None = None,
    detector: FaceDetector | None = None,
    preprocessor: Preprocessor | None = None,
    estimator: GazeEstimator | None = None,
    bus: EventBus | None = None,
    clock: Clock | None = None,
    calibration_model: CalibrationModel | None = None,
    calibration_profile_id: str | None = None,
    mapper: ScreenMapper | None = None,
    filter_chain: Filter | None = None,
    cursor: CursorController | None = None,
    screen: ScreenInfo | None = None,
    warmup: bool = True,
) -> PipelineComponents:
    """Construct a :class:`PipelineComponents` from ``cfg`` + overrides.

    Args:
        cfg: The full application config. Every section is consulted.
        source, detector, preprocessor, estimator, bus, clock: Optional
            substitutes (Sprint 7).
        calibration_model: Explicit calibration transform. If ``None``
            and ``calibration_profile_id`` is provided, the profile is
            loaded from disk. Otherwise no calibration is wired and the
            runner uses raw model output as calibrated (with a warning).
        calibration_profile_id: Name of a persisted profile in
            ``cfg.calibration.profiles_dir``.
        mapper: Explicit :class:`~irisflow.core.interfaces.ScreenMapper`.
            When ``None`` a default one is built from ``cfg.mapping`` +
            ``screen``.
        filter_chain: Pre-built filter chain. When ``None`` a chain is
            built from ``cfg.filtering``.
        cursor: Cursor controller. When ``None`` a
            :class:`~irisflow.control.noop.NoOpCursor` is wired — the
            safe default.
        screen: :class:`ScreenInfo` describing the target display.
            Required when ``cfg.control.enabled`` is True and ``mapper``
            is not supplied explicitly (the CLI queries the OS). Tests
            typically supply an explicit :class:`ScreenInfo`.
        warmup: When true and the estimator was built here (not injected),
            call :func:`warm_up_backend` before returning.
    """
    real_clock: Clock = clock if clock is not None else SystemClock()
    real_bus = bus if bus is not None else EventBus()
    real_source = source if source is not None else _default_source(cfg, real_clock)
    real_detector = detector if detector is not None else _default_detector(cfg)
    real_preprocessor = preprocessor if preprocessor is not None else _default_preprocessor(cfg)
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

    real_calibration = calibration_model
    if real_calibration is None and calibration_profile_id is not None:
        real_calibration = _load_calibration_profile(cfg, calibration_profile_id)

    real_mapper: ScreenMapper | None = mapper
    if real_mapper is None and screen is not None:
        real_mapper = ScreenPixelMapper(screen=screen, clamp_margin_px=cfg.mapping.clamp_margin_px)

    real_filter: Filter | None = filter_chain
    if real_filter is None:
        real_filter = build_filter_chain(filter_config_from_app(cfg))

    real_cursor: CursorController = cursor if cursor is not None else NoOpCursor(enabled=False)

    return PipelineComponents(
        source=real_source,
        detector=real_detector,
        preprocessor=real_preprocessor,
        estimator=real_estimator,
        bus=real_bus,
        state=PipelineStateMachine(real_bus, clock=real_clock),
        metrics=MetricsRecorder(_clock=real_clock),
        clock=real_clock,
        calibration_model=real_calibration,
        mapper=real_mapper,
        filter_chain=real_filter,
        cursor=real_cursor,
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
    inner = MediaPipeFaceDetector(mesh, roi_margin=cfg.detection.roi_margin, square_rois=True)
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


def filter_config_from_app(cfg: AppConfig) -> ChainConfig:
    """Translate the :class:`FilterConfig` (Pydantic) to a plain :class:`ChainConfig`.

    The filtering module lives in a sibling layer to :mod:`irisflow.config`,
    so it must not import the Pydantic types directly (import-linter
    contract). The orchestrator owns the translation.
    """
    # Filter chain accepts only the four names defined in filtering.chain;
    # AppConfig's ``FilterName`` allows the same values plus "kalman".
    supported: set[str] = {"outlier", "ema", "one_euro", "fixation"}
    filtered_names: list[FilterName] = []
    for name in cfg.filtering.chain:
        if name in supported:
            filtered_names.append(name)  # type: ignore[arg-type]
    chain_names: tuple[FilterName, ...] = tuple(filtered_names)
    if not chain_names:
        chain_names = ("outlier", "one_euro", "fixation")
    return ChainConfig(
        chain=chain_names,
        outlier=OutlierParams(max_velocity_px_per_s=cfg.filtering.outlier.max_velocity_px_per_s),
        ema=EmaParams(alpha=cfg.filtering.ema.alpha),
        one_euro=OneEuroParams(
            min_cutoff=cfg.filtering.one_euro.min_cutoff,
            beta=cfg.filtering.one_euro.beta,
            d_cutoff=cfg.filtering.one_euro.d_cutoff,
        ),
        fixation=FixationParams(
            velocity_threshold_px_per_s=cfg.filtering.fixation.velocity_threshold_px_per_s
        ),
    )


def _load_calibration_profile(cfg: AppConfig, profile_id: str) -> CalibrationModel:
    """Load a persisted profile, or fall back to a fresh (unfitted) model."""
    store = CalibrationStore(cfg.calibration.profiles_dir)
    try:
        profile = store.load(profile_id)
    except CalibrationError:
        # Return an unfitted default; the runner reports "not calibrated"
        # via the SafetyGate/logger. We do not hard-fail because the
        # ALS user must still be able to move the mouse via raw gaze.
        return build_calibration_model(
            cfg.calibration.model_kind,
            profile_id=profile_id,
            polynomial_degree=cfg.calibration.polynomial_degree,
        )
    return profile.model
