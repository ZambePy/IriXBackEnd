"""Sprint 11 E2E — grava uma sessão real do pipeline e replaya bit-idêntica.

The end-to-end guarantee (SPRINTS §11 DoD):

* Boot a pipeline with only stubs (no camera, no model);
* run the recorder as a bus sink;
* stop everything, read the JSONL back, feed it through
  :class:`SessionReplayer`;
* every downstream event (:class:`GazeUpdated`, :class:`DwellProgress`,
  :class:`DwellClick`, safety events) must match the recorded ones
  element-by-element.

If this ever fails, the replay determinism promise is broken and
Sprint 11 has regressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.safety import RestZone, SafetyGate
from irisflow.core.clock import Clock, FakeClock
from irisflow.core.events import (
    DwellClick,
    DwellProgress,
    GazeUpdated,
    SafetyPaused,
    SafetyResumed,
)
from irisflow.filtering.chain import (
    ChainConfig,
    FixationParams,
    OneEuroParams,
    OutlierParams,
    build_filter_chain,
)
from irisflow.mapping.screen import ScreenMapper as ScreenPixelMapper
from irisflow.mapping.screen_info import ScreenInfo
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.control_sink import ControlSink
from irisflow.pipeline.orchestrator import PipelineComponents
from irisflow.pipeline.replayer import SessionReplayer
from irisflow.pipeline.runner import PipelineRunner
from irisflow.pipeline.state import PipelineStateMachine
from irisflow.telemetry.metrics import MetricsRecorder
from irisflow.telemetry.recorder import SessionRecorder
from irisflow.telemetry.report import build_session_report
from irisflow.telemetry.session import SessionHeader, read_session_file
from tests.fixtures.stubs import (
    IdentityScreenMapper,
    PassthroughPreprocessor,
    StubFaceDetector,
    StubGazeEstimator,
)


class _AutoAdvanceClock:
    __slots__ = ("_now", "_step")

    def __init__(self, *, step_s: float = 0.05) -> None:
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
class _NoopCursor:
    _enabled: bool = False
    moves: list[tuple[int, int]] = field(default_factory=list)
    clicks: list[str] = field(default_factory=list)

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


SCREEN_W = 1000
SCREEN_H = 800
DWELL = DwellParams(radius_px=200, duration_ms=100, refractory_ms=40)
CHAIN = ChainConfig(
    chain=("outlier", "one_euro", "fixation"),
    outlier=OutlierParams(max_velocity_px_per_s=100_000.0),
    one_euro=OneEuroParams(min_cutoff=5.0, beta=0.3, d_cutoff=5.0),
    fixation=FixationParams(velocity_threshold_px_per_s=40.0),
)


def _build_components(clock: Clock, cursor: _NoopCursor) -> PipelineComponents:
    bus = EventBus()
    return PipelineComponents(
        source=SyntheticFrameSource(width=64, height=48, clock=clock),
        detector=StubFaceDetector(),
        preprocessor=PassthroughPreprocessor(),
        estimator=StubGazeEstimator(trajectory=[(0.5, 0.5)]),
        bus=bus,
        state=PipelineStateMachine(bus, clock=clock),
        metrics=MetricsRecorder(_clock=clock),
        clock=clock,
        mapper=IdentityScreenMapper(screen_width=SCREEN_W, screen_height=SCREEN_H),
        filter_chain=build_filter_chain(CHAIN),
        cursor=cursor,
    )


def _build_control_sink(components: PipelineComponents, clock: Clock) -> ControlSink:
    return ControlSink(
        bus=components.bus,
        cursor=components.cursor,  # type: ignore[arg-type]
        safety=SafetyGate(
            pause_on_face_lost_ms=1000,
            rest_zone=RestZone(0, 0, 0, 0),
            clock=clock,
        ),
        dwell=DwellClicker(DWELL),
        cursor_enabled_on_start=True,
        clock=clock,
    )


def _build_replayer() -> SessionReplayer:
    clock_replay = FakeClock()
    return SessionReplayer(
        calibration_model=None,
        mapper=ScreenPixelMapper(
            screen=ScreenInfo(width_px=SCREEN_W, height_px=SCREEN_H),
            clamp_margin_px=0,
        ),
        filter_chain=build_filter_chain(CHAIN),
        safety=SafetyGate(
            pause_on_face_lost_ms=1000,
            rest_zone=RestZone(0, 0, 0, 0),
            clock=clock_replay,
        ),
        dwell=DwellClicker(DWELL),
        cursor_enabled=True,
        clock=clock_replay,
    )


def test_record_then_replay_matches_bit_identical(tmp_path: Path) -> None:
    clock = _AutoAdvanceClock()
    cursor = _NoopCursor()
    components = _build_components(clock, cursor)
    sink = _build_control_sink(components, clock)

    header = SessionHeader(
        session_id="e2e-record-replay",
        wall_start_s=0.0,
        screen_width_px=SCREEN_W,
        screen_height_px=SCREEN_H,
        calibration_profile_id=None,
    )
    recorder = SessionRecorder(
        output_path=tmp_path / "session.jsonl",
        header=header,
        subscribe=components.bus.subscribe,
        unsubscribe=components.bus.unsubscribe,
        clock=clock,
    )

    sink.start()
    recorder.start()
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=25)
    recorder.stop()
    sink.stop()

    _, recorded_events = read_session_file(recorder.output_path)

    recorded_gaze = [e for e in recorded_events if isinstance(e, GazeUpdated)]
    recorded_progress = [e for e in recorded_events if isinstance(e, DwellProgress)]
    recorded_clicks = [e for e in recorded_events if isinstance(e, DwellClick)]
    recorded_paused = [e for e in recorded_events if isinstance(e, SafetyPaused)]
    recorded_resumed = [e for e in recorded_events if isinstance(e, SafetyResumed)]

    assert recorded_gaze, "expected the runner to produce gaze updates"

    replay_result = _build_replayer().replay(recorded_events)

    assert replay_result.gaze_updates == recorded_gaze
    assert replay_result.dwell_progress == recorded_progress
    assert replay_result.dwell_clicks == recorded_clicks
    assert replay_result.safety_paused == recorded_paused
    assert replay_result.safety_resumed == recorded_resumed


def test_report_generated_from_recorded_session(tmp_path: Path) -> None:
    clock = _AutoAdvanceClock()
    cursor = _NoopCursor()
    components = _build_components(clock, cursor)
    sink = _build_control_sink(components, clock)
    recorder = SessionRecorder(
        output_path=tmp_path / "session.jsonl",
        header=SessionHeader(
            session_id="report-test",
            wall_start_s=0.0,
            screen_width_px=SCREEN_W,
            screen_height_px=SCREEN_H,
            calibration_profile_id=None,
        ),
        subscribe=components.bus.subscribe,
        unsubscribe=components.bus.unsubscribe,
    )
    sink.start()
    recorder.start()
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=20)
    recorder.stop()
    sink.stop()

    header, events = read_session_file(recorder.output_path)
    report = build_session_report(session_id=header.session_id, events=events)
    assert report.frames_ok > 0
    assert report.duration_s > 0
    assert report.fps > 0
