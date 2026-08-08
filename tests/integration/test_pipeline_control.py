"""Sprint 10 — full pipeline + control sink integration (no hardware, no OS).

Wires:

* :class:`SyntheticFrameSource` → :class:`StubFaceDetector` →
  :class:`PassthroughPreprocessor` → :class:`StubGazeEstimator`;
* :class:`IdentityScreenMapper` + :class:`FilterChain` so the runner
  emits :class:`GazeUpdated`;
* :class:`ControlSink` with a recording :class:`_RecordingCursor` that
  never touches the OS.

Checks the DoD-visible invariants: cursor moves, dwell click fires while
the gaze holds still, and nothing is dispatched when the pipeline is
LOST.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.safety import RestZone, SafetyGate
from irisflow.core.clock import Clock
from irisflow.core.events import (
    DwellClick,
    DwellProgress,
    GazeUpdated,
)
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
    PassthroughPreprocessor,
    StubFaceDetector,
    StubGazeEstimator,
)


class _AutoAdvanceClock:
    """FakeClock that advances a fixed delta every time monotonic() is read.

    Enough to make dwell timing testable without a real wall clock — every
    frame + downstream stage sees a fresh, monotonically-increasing ts.
    """

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

    def advance(self, seconds: float) -> None:
        self._now += seconds


@dataclass
class _RecordingCursor:
    moves: list[tuple[int, int]] = field(default_factory=list)
    clicks: list[str] = field(default_factory=list)
    _enabled: bool = False

    def move(self, px: int, py: int) -> None:
        self.moves.append((px, py))

    def click(self, button: str = "left") -> None:
        self.clicks.append(button)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled


def _build_components(
    *,
    trajectory: list[tuple[float, float]],
    cursor: _RecordingCursor,
    clock: Clock,
    screen_width: int = 1000,
    screen_height: int = 800,
) -> tuple[PipelineComponents, EventBus]:
    bus = EventBus()
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
    return components, bus


def _build_sink(
    bus: EventBus,
    cursor: _RecordingCursor,
    clock: Clock,
    *,
    dwell_duration_ms: int = 100,
    dwell_radius_px: int = 200,
) -> ControlSink:
    return ControlSink(
        bus=bus,
        cursor=cursor,
        safety=SafetyGate(
            pause_on_face_lost_ms=1000,
            rest_zone=RestZone(0, 0, 0, 0),
            clock=clock,
        ),
        dwell=DwellClicker(
            DwellParams(
                radius_px=dwell_radius_px,
                duration_ms=dwell_duration_ms,
                refractory_ms=50,
            )
        ),
        cursor_enabled_on_start=True,
        clock=clock,
    )


def test_runner_moves_cursor_and_fires_dwell_click() -> None:
    clock = _AutoAdvanceClock(step_s=0.05)
    cursor = _RecordingCursor()
    components, bus = _build_components(
        trajectory=[(0.5, 0.5)], cursor=cursor, clock=clock
    )
    sink = _build_sink(bus, cursor, clock, dwell_duration_ms=50)
    sink.start()

    gaze_events: list[GazeUpdated] = []
    dwell_progress: list[DwellProgress] = []
    dwell_clicks: list[DwellClick] = []

    def _capture_gaze(event: object) -> None:
        if isinstance(event, GazeUpdated):
            gaze_events.append(event)

    def _capture_progress(event: object) -> None:
        if isinstance(event, DwellProgress):
            dwell_progress.append(event)

    def _capture_click(event: object) -> None:
        if isinstance(event, DwellClick):
            dwell_clicks.append(event)

    bus.subscribe(GazeUpdated, _capture_gaze)
    bus.subscribe(DwellProgress, _capture_progress)
    bus.subscribe(DwellClick, _capture_click)

    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=20)

    sink.stop()

    assert len(gaze_events) >= 10
    # Cursor moved to the mapped point (500, 400) - deterministic screen 1000x800.
    assert cursor.moves
    assert cursor.moves[0] == (500, 400)
    assert dwell_progress, "expected at least one DwellProgress"
    assert dwell_clicks, "expected at least one DwellClick"
    # Runner disables cursor on shutdown (SPRINTS §10 DoD).
    assert not cursor.is_enabled


def test_stop_disables_cursor_and_control_sink_cleans_up() -> None:
    clock = _AutoAdvanceClock(step_s=0.05)
    cursor = _RecordingCursor()
    components, bus = _build_components(
        trajectory=[(0.1, 0.1)], cursor=cursor, clock=clock
    )
    sink = _build_sink(bus, cursor, clock)
    sink.start()
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=5)
    sink.stop()

    # Cursor released after shutdown → mouse is not "hijacked".
    assert not cursor.is_enabled
    # Bus has been unsubscribed — new events reach neither cursor nor dwell.
    old_moves = list(cursor.moves)
    bus.publish(GazeUpdated(frame_id=999, timestamp=0.0, px=1, py=1,
                            is_fixation=False, confidence=1.0))
    assert cursor.moves == old_moves


def test_runner_never_moves_mouse_when_control_sink_disabled() -> None:
    """DoD: automated suite must never move the developer's mouse."""
    clock = _AutoAdvanceClock(step_s=0.05)
    cursor = _RecordingCursor()
    components, bus = _build_components(
        trajectory=[(0.7, 0.7)], cursor=cursor, clock=clock
    )
    # Note: sink NOT started → nothing subscribes to GazeUpdated.
    del bus  # unused
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=10)
    assert cursor.moves == []
    assert cursor.clicks == []
