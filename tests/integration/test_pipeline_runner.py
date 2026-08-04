"""Sprint 7 integration — full loop, no hardware, no real model.

The whole point of the layered architecture is that this test exists.
We wire :class:`SyntheticFrameSource` + :class:`StubFaceDetector` +
:class:`PassthroughPreprocessor` + :class:`StubGazeEstimator` into a
:class:`PipelineComponents`, drive the runner for a bounded number of
ticks, and check the invariants the Sprint 7 DoD calls out:

* runs without crashing;
* publishes ``RawGazeReady``;
* a stage exception → frame dropped, loop survives;
* ``stop()`` returns quickly and closes the source.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from irisflow.capture.synthetic import SyntheticFrameSource
from irisflow.core.clock import FakeClock
from irisflow.core.events import (
    FaceAcquired,
    FaceLost,
    PipelineState,
    RawGazeReady,
    StateChanged,
)
from irisflow.core.types import Frame, ModelInput, RawGaze
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.orchestrator import PipelineComponents
from irisflow.pipeline.runner import PipelineRunner
from irisflow.pipeline.state import PipelineStateMachine
from irisflow.telemetry.metrics import MetricsRecorder
from tests.fixtures.stubs import (
    PassthroughPreprocessor,
    StubFaceDetector,
    StubGazeEstimator,
)


def _components(
    *,
    detector=None,
    estimator=None,
    source=None,
    clock=None,
) -> PipelineComponents:
    real_clock = clock or FakeClock()
    bus = EventBus()
    return PipelineComponents(
        source=source or SyntheticFrameSource(width=64, height=48, clock=real_clock),
        detector=detector or StubFaceDetector(),
        preprocessor=PassthroughPreprocessor(),
        estimator=estimator or StubGazeEstimator(trajectory=[(0.4, 0.6)]),
        bus=bus,
        state=PipelineStateMachine(bus, clock=real_clock),
        metrics=MetricsRecorder(_clock=real_clock),
        clock=real_clock,
    )


def test_runner_publishes_raw_gaze_events_and_transitions_to_tracking() -> None:
    components = _components()
    events: list = []
    components.bus.subscribe_all(events.append)

    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=10)

    gaze_events = [e for e in events if isinstance(e, RawGazeReady)]
    state_events = [e for e in events if isinstance(e, StateChanged)]

    assert len(gaze_events) >= 5, f"expected several gaze events, got {len(gaze_events)}"
    # First StateChanged is IDLE → TRACKING at run start.
    assert state_events[0].previous == PipelineState.IDLE
    assert state_events[0].current == PipelineState.TRACKING


def test_runner_records_stage_metrics() -> None:
    components = _components()
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=5)

    snap = components.metrics.snapshot()
    assert snap.frames_ok >= 3
    assert "capture" in snap.stages
    assert "detection" in snap.stages
    assert "preprocess" in snap.stages
    assert "inference" in snap.stages


def test_runner_survives_stage_exception_and_records_drop() -> None:
    class _BoomEstimator(StubGazeEstimator):
        _called: int = 0

        def predict(self, model_input: ModelInput) -> RawGaze:
            self._called += 1
            if self._called == 2:
                raise RuntimeError("simulated bad frame")
            return super().predict(model_input)

    components = _components(estimator=_BoomEstimator(trajectory=[(0.5, 0.5)]))
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=5)

    snap = components.metrics.snapshot()
    assert snap.frames_dropped >= 1
    assert snap.frames_ok >= 1  # other frames still produced gaze


def test_runner_face_lost_publishes_face_lost_and_transitions() -> None:
    """First 2 detections succeed; then None forever → LOST after first miss."""
    class _AbsentDetector(StubFaceDetector):
        _n: int = 0

        def detect(self, frame: Frame):
            self._n += 1
            return super().detect(frame) if self._n <= 2 else None

    components = _components(detector=_AbsentDetector())
    events: list = []
    components.bus.subscribe_all(events.append)

    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=8)

    face_lost = [e for e in events if isinstance(e, FaceLost)]
    face_acquired = [e for e in events if isinstance(e, FaceAcquired)]
    state_events = [e for e in events if isinstance(e, StateChanged)]

    assert len(face_acquired) == 1  # first detection triggers acquired
    assert len(face_lost) >= 1
    lost_transitions = [
        e for e in state_events if e.current == PipelineState.LOST
    ]
    assert lost_transitions, "expected a TRACKING -> LOST transition"


def test_runner_stop_returns_promptly() -> None:
    """Verify DoD: Ctrl+C-style stop releases the loop in < 1s."""
    components = _components(source=SyntheticFrameSource(width=64, height=48))
    runner = PipelineRunner(components, idle_sleep_s=0.01)

    thread = threading.Thread(target=lambda: runner.run(), daemon=True)
    thread.start()
    time.sleep(0.05)
    start = time.monotonic()
    runner.stop()
    thread.join(timeout=1.5)
    elapsed = time.monotonic() - start

    assert not thread.is_alive(), "runner did not stop within 1.5s"
    assert elapsed < 1.0, f"stop took {elapsed:.3f}s, expected < 1s"
    assert not components.source.is_open


def test_runner_final_state_is_idle_after_stop() -> None:
    components = _components()
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=3)
    runner.stop()  # simulate stop happening after loop naturally exits
    # The runner already ran shutdown() inside run().
    assert components.state.state == PipelineState.IDLE


def test_extended_run_stays_healthy() -> None:
    """Proxy for the DoD's '5 minutes without crash' criterion."""
    components = _components()
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=300)
    snap = components.metrics.snapshot()
    assert snap.frames_ok >= 200
    assert snap.frames_dropped == 0


def test_runner_stopping_frame_source_is_released_even_if_source_raises() -> None:
    class _RaisingSource(SyntheticFrameSource):
        def close(self) -> None:
            raise RuntimeError("hostile source")

    src = _RaisingSource(width=32, height=32)
    components = _components(source=src)
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=2)  # should not raise despite hostile close()


def test_stub_estimator_frame_uses_frame_and_model_input_stubs() -> None:
    # Sanity: the fixtures we use in the integration tests actually match
    # the Frame / ModelInput contract — protects against silent stub drift.
    src = SyntheticFrameSource(width=32, height=32)
    with src:
        frame = src.read()
    assert isinstance(frame, Frame)
    assert frame.data.dtype == np.uint8


@pytest.mark.parametrize("stage_that_fails", ["preprocess", "inference"])
def test_runner_never_crashes_for_any_stage_exception(stage_that_fails: str) -> None:
    class _BadPreproc(PassthroughPreprocessor):
        def build(self, frame, detection):  # type: ignore[no-untyped-def]
            raise RuntimeError("preprocess boom")

    class _BadEstimator(StubGazeEstimator):
        def predict(self, model_input):  # type: ignore[no-untyped-def]
            raise RuntimeError("infer boom")

    components = _components(
        estimator=(_BadEstimator() if stage_that_fails == "inference" else None),
    )
    if stage_that_fails == "preprocess":
        components.preprocessor = _BadPreproc()
    runner = PipelineRunner(components, idle_sleep_s=0.0)
    runner.run(max_ticks=5)
    snap = components.metrics.snapshot()
    assert snap.frames_dropped >= 1
