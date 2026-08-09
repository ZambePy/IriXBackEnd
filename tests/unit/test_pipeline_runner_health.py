"""Tests for runner gaze health diagnostics (T11)."""
from __future__ import annotations

import pytest

from irisflow.core.clock import FakeClock
from irisflow.pipeline.bus import EventBus
from irisflow.pipeline.orchestrator import PipelineComponents
from irisflow.pipeline.runner import PipelineRunner
from irisflow.pipeline.state import PipelineStateMachine
from irisflow.telemetry.metrics import MetricsRecorder
from tests.fixtures.stubs import (
    IdentityCalibrationModel,
    IdentityFilter,
    IdentityScreenMapper,
    PassthroughPreprocessor,
    StubFaceDetector,
    StubGazeEstimator,
    SyntheticFrameSource,
)


def _make_components(
    trajectory: list[tuple[float, float]],
    clock: FakeClock | None = None,
) -> PipelineComponents:
    if clock is None:
        clock = FakeClock()
    bus = EventBus()
    source = SyntheticFrameSource()
    return PipelineComponents(
        source=source,
        detector=StubFaceDetector(),
        preprocessor=PassthroughPreprocessor(),
        estimator=StubGazeEstimator(trajectory=trajectory),
        bus=bus,
        state=PipelineStateMachine(bus, clock=clock),
        metrics=MetricsRecorder(_clock=clock),
        clock=clock,
        calibration_model=IdentityCalibrationModel(),
        mapper=IdentityScreenMapper(),
        filter_chain=IdentityFilter(),
    )


def test_saturation_warning_emitted(capsys: pytest.CaptureFixture[str]) -> None:
    """Warn when raw_x or raw_y is stuck at 0.0 for >= SAT_WARN_FRAMES frames."""
    n = PipelineRunner._SAT_WARN_FRAMES + 5
    trajectory = [(0.0, 0.5)] * n  # raw_x saturated at 0.0
    components = _make_components(trajectory)
    runner = PipelineRunner(components)
    runner.run(max_ticks=n + 2)
    captured = capsys.readouterr()
    assert "saturated" in captured.out, (
        "Expected gaze.saturated warning not found in output"
    )


def test_no_saturation_warning_when_gaze_moves(capsys: pytest.CaptureFixture[str]) -> None:
    """No saturation warning when gaze values vary normally."""
    trajectory = [(i * 0.01, 0.5) for i in range(50)]
    components = _make_components(trajectory)
    runner = PipelineRunner(components)
    runner.run(max_ticks=52)
    captured = capsys.readouterr()
    assert "saturated" not in captured.out, (
        f"Unexpected saturation warning in output: {captured.out}"
    )


def test_degeneracy_warning_emitted(capsys: pytest.CaptureFixture[str]) -> None:
    """Warn when gaze amplitude is very low over a 5-second window."""
    # All samples at exactly (0.5, 0.5) — zero amplitude.
    # Use fps=1.0 so frame timestamps advance 1 second per frame,
    # ensuring the 5-second degeneracy window fills within 100 ticks.
    trajectory = [(0.5, 0.5)] * 200
    clock = FakeClock()
    bus = EventBus()
    source = SyntheticFrameSource(fps=1.0)  # 1 frame/s → timestamps 0, 1, 2, ...
    components = PipelineComponents(
        source=source,
        detector=StubFaceDetector(),
        preprocessor=PassthroughPreprocessor(),
        estimator=StubGazeEstimator(trajectory=trajectory),
        bus=bus,
        state=PipelineStateMachine(bus, clock=clock),
        metrics=MetricsRecorder(_clock=clock),
        clock=clock,
        calibration_model=IdentityCalibrationModel(),
        mapper=IdentityScreenMapper(),
        filter_chain=IdentityFilter(),
    )
    runner = PipelineRunner(components)
    runner.run(max_ticks=100)
    captured = capsys.readouterr()
    assert "degenerate" in captured.out, (
        "Expected gaze.degenerate warning not found in output"
    )
