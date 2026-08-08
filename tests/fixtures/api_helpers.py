"""Test helpers for building an :class:`AppState` with only stubs.

The real :func:`build_app_state_from_config` opens a webcam and loads the
Keras model — neither is available in CI. These helpers wire the same
collaborators from :mod:`tests.fixtures.stubs`.
"""

from __future__ import annotations

from dataclasses import dataclass

from irisflow.api.session import SessionHub
from irisflow.api.state import AppState
from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.config.schema import AppConfig
from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.safety import RestZone, SafetyGate
from irisflow.core.clock import Clock
from irisflow.filtering.chain import (
    ChainConfig,
    FixationParams,
    OneEuroParams,
    OutlierParams,
    build_filter_chain,
)
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.control_sink import ControlSink
from irisflow.pipeline.orchestrator import PipelineComponents
from irisflow.pipeline.runner import PipelineRunner
from irisflow.pipeline.state import PipelineStateMachine
from irisflow.telemetry.metrics import MetricsRecorder
from tests.fixtures.stubs import (
    IdentityScreenMapper,
    NoOpCursorController,
    PassthroughPreprocessor,
    StubFaceDetector,
    StubGazeEstimator,
)


class AutoAdvanceClock:
    """FakeClock that advances a fixed delta every monotonic() read."""

    __slots__ = ("_now", "_step")

    def __init__(self, *, step_s: float = 0.02) -> None:
        self._now = 0.0
        self._step = step_s

    def monotonic(self) -> float:
        self._now += self._step
        return self._now

    def wall(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now += seconds


@dataclass
class StubbedApp:
    """Bundle of stub-wired collaborators for the api tests."""

    state: AppState
    bus: EventBus
    cursor: NoOpCursorController
    components: PipelineComponents


def build_stubbed_app_state(
    *,
    config: AppConfig,
    trajectory: list[tuple[float, float]] | None = None,
    screen_width: int = 1000,
    screen_height: int = 800,
    step_s: float = 0.02,
) -> StubbedApp:
    """Return an :class:`AppState` wired entirely with stubs."""
    if trajectory is None:
        trajectory = [(0.5, 0.5)]
    clock: Clock = AutoAdvanceClock(step_s=step_s)
    bus = EventBus()
    cursor = NoOpCursorController()
    chain = build_filter_chain(
        ChainConfig(
            chain=("outlier", "one_euro", "fixation"),
            outlier=OutlierParams(max_velocity_px_per_s=100_000.0),
            one_euro=OneEuroParams(min_cutoff=10.0, beta=0.5, d_cutoff=10.0),
            fixation=FixationParams(velocity_threshold_px_per_s=50.0),
        )
    )
    components = PipelineComponents(
        source=SyntheticFrameSource(width=64, height=48, clock=clock),
        detector=StubFaceDetector(),
        preprocessor=PassthroughPreprocessor(),
        estimator=StubGazeEstimator(trajectory=trajectory),
        bus=bus,
        state=PipelineStateMachine(bus, clock=clock),
        metrics=MetricsRecorder(_clock=clock),
        clock=clock,
        mapper=IdentityScreenMapper(
            screen_width=screen_width, screen_height=screen_height
        ),
        filter_chain=chain,
        cursor=cursor,
    )
    control_sink = ControlSink(
        bus=bus,
        cursor=cursor,
        safety=SafetyGate(
            pause_on_face_lost_ms=1000,
            rest_zone=RestZone(0, 0, 0, 0),
            clock=clock,
        ),
        dwell=DwellClicker(
            DwellParams(radius_px=200, duration_ms=100, refractory_ms=50)
        ),
        cursor_enabled_on_start=False,
        clock=clock,
    )
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    hub = SessionHub(bus, screen_width=screen_width, screen_height=screen_height)
    state = AppState(
        config=config,
        components=components,
        control_sink=control_sink,
        runner=runner,
        hub=hub,
        screen_width=screen_width,
        screen_height=screen_height,
    )
    return StubbedApp(state=state, bus=bus, cursor=cursor, components=components)
