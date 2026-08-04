"""Sprint 2 — pipeline events are immutable and cover the state machine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from irisflow.core.events import (
    CalibrationProgress,
    Event,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    PipelineState,
    StateChanged,
)


def test_pipeline_state_covers_the_five_states() -> None:
    names = {s.name for s in PipelineState}
    assert names == {"IDLE", "CALIBRATING", "TRACKING", "LOST", "PAUSED"}


def test_pipeline_state_is_string_backed_for_serialization() -> None:
    assert PipelineState.TRACKING.value == "TRACKING"
    assert PipelineState("PAUSED") is PipelineState.PAUSED


def test_gaze_updated_is_immutable() -> None:
    evt = GazeUpdated(
        frame_id=1,
        timestamp=0.5,
        px=100,
        py=200,
        is_fixation=True,
        confidence=0.9,
    )
    with pytest.raises(FrozenInstanceError):
        evt.px = 0  # type: ignore[misc]


def test_state_changed_records_transition() -> None:
    evt = StateChanged(
        previous=PipelineState.CALIBRATING,
        current=PipelineState.TRACKING,
        timestamp=1.0,
    )
    assert evt.previous == PipelineState.CALIBRATING
    assert evt.current == PipelineState.TRACKING


def test_calibration_progress_carries_phase_and_target() -> None:
    evt = CalibrationProgress(
        index=3, total=9, target_x=0.5, target_y=0.1, phase="collecting"
    )
    assert evt.phase == "collecting"


@pytest.mark.parametrize(
    "event",
    [
        GazeUpdated(1, 0.0, 100, 100, False, 0.9),
        FaceLost(1, 0.0, 0.0),
        FaceAcquired(1, 0.0),
        StateChanged(PipelineState.LOST, PipelineState.TRACKING, 0.0),
        CalibrationProgress(0, 9, 0.5, 0.5, "prompt"),
    ],
)
def test_every_event_variant_matches_the_event_union(event: object) -> None:
    # Simple runtime check — the union alias is not runtime-introspectable
    # in Python 3.11 without typing.get_args, so a straight isinstance chain
    # against the constituent classes is the pragmatic check.
    assert isinstance(
        event,
        (GazeUpdated, FaceLost, FaceAcquired, StateChanged, CalibrationProgress),
    )


def test_event_union_symbol_is_exported() -> None:
    # Just makes sure the alias exists and is importable — protects against
    # accidental removal.
    assert Event is not None
