"""Sprint 11 — recorder overhead is well under the 3% budget.

The SPRINTS §11 DoD calls out "overhead ≤ 3% of frame time". At 30 FPS
that means the recorder must not spend more than ~1 ms per frame. Even
a slow local disk finishes an appended JSONL line in a few tens of
microseconds — this test asserts a comfortable margin.

The test is intentionally soft (upper bound of 1 ms per event, not 100
µs) so it doesn't flake on shared CI runners.
"""

from __future__ import annotations

import time
from pathlib import Path

from irisflow.core.events import RawGazeReady
from irisflow.pipeline.bus import EventBus
from irisflow.telemetry.recorder import SessionRecorder
from irisflow.telemetry.session import SessionHeader


def test_recorder_per_event_write_under_one_ms(tmp_path: Path) -> None:
    bus = EventBus()
    header = SessionHeader(
        session_id="overhead",
        wall_start_s=0.0,
        screen_width_px=1000,
        screen_height_px=800,
        calibration_profile_id=None,
    )
    recorder = SessionRecorder(
        output_path=tmp_path / "overhead.jsonl",
        header=header,
        subscribe=bus.subscribe,
        unsubscribe=bus.unsubscribe,
    )
    recorder.start()
    try:
        n = 500
        start = time.perf_counter()
        for i in range(n):
            bus.publish(
                RawGazeReady(
                    frame_id=i,
                    timestamp=i * 0.033,
                    x=0.5,
                    y=0.5,
                    confidence=1.0,
                    inference_ms=0.0,
                )
            )
        elapsed = time.perf_counter() - start
    finally:
        recorder.stop()

    per_event_ms = (elapsed / n) * 1000.0
    # 1 ms / event corresponds to ~3% of a 33 ms frame at 30 FPS.
    assert per_event_ms < 1.0, f"recorder overhead {per_event_ms:.3f} ms/event"
    assert recorder.events_written == n
