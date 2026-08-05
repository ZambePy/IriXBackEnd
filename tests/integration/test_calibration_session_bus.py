"""Sprint 8 -- session events reach the real EventBus.

Verifies the layer seam: session publishes CalibrationProgress via a
plain callable (bus.publish), the bus fans them out to subscribers.
No pipeline runner involved -- we just drive the session by hand.
"""

from __future__ import annotations

import numpy as np

from irisflow.calibration.session import CalibrationSession
from irisflow.core.clock import FakeClock
from irisflow.core.events import CalibrationProgress
from irisflow.core.types import RawGaze
from irisflow.pipeline.bus import EventBus


def test_bus_delivers_progress_events_from_session() -> None:
    bus = EventBus()
    received: list[CalibrationProgress] = []
    bus.subscribe(CalibrationProgress, lambda e: received.append(e))  # type: ignore[arg-type]

    clock = FakeClock()
    session = CalibrationSession(
        profile_id="int-test",
        screen_width=1600,
        screen_height=900,
        point_count=5,
        samples_per_point=4,
        stabilization_ms=0,
        clock=clock,
        publisher=bus.publish,  # <-- the layer seam
    )
    session.start()
    rng = np.random.default_rng(0)
    for target in session.targets:
        session.begin_target(target.index)
        clock.advance(0.05)
        iters = 0
        while not session.is_current_target_done and iters < 200:
            noise = float(rng.normal(0.0, 0.001))
            session.push_sample(
                RawGaze(
                    x=target.nx + noise,
                    y=target.ny + noise,
                    confidence=1.0,
                    inference_ms=1.0,
                ),
                clock.monotonic(),
            )
            clock.advance(0.033)
            iters += 1
    session.finalize()

    phases_seen = {e.phase for e in received}
    assert "prompt" in phases_seen
    assert "settled" in phases_seen
    assert "done" in phases_seen
    # every target announced at least once
    indices = {e.index for e in received}
    assert indices == {t.index for t in session.targets}
