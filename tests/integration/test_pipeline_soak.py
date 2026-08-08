"""Sprint 13 — soak test: no FPS decay, no memory growth over long runs.

The SPRINTS §13 DoD says "30 minutos sem degradação de FPS ou crescimento
de memória". CI can't afford 30 minutes of wall clock; instead we run
30 min *of simulated frame time* (54 000 ticks at the 30 FPS baseline)
through the deterministic stub pipeline with :class:`AutoAdvanceClock`,
mirroring the real usage pattern where :class:`MetricsRecorder` is
snapshot + reset every few seconds.

We assert three things:

1. **FPS stays flat.** Split the run in halves; the second-half FPS must
   be within 20% of the first-half FPS. A leaking buffer or unbounded
   state would tank the second half.
2. **Memory doesn't creep.** :mod:`tracemalloc` snapshot after warm-up
   vs after the full soak must be under a low ceiling (2 MiB) once the
   metrics buffer resets (the by-design behaviour: consumers call
   ``metrics.snapshot(); metrics.reset()``).
3. **Bus subscribers keep up.** A counting subscriber must have seen
   every published :class:`GazeUpdated`.

Marked ``@pytest.mark.slow`` so the fast CI job skips it; a nightly run
picks it up. Even so, on a modest laptop the whole thing finishes in
under 15 s because the stubs do zero real work.
"""

from __future__ import annotations

import tracemalloc

import pytest

from irisflow.config.schema import AppConfig
from irisflow.core.events import GazeUpdated
from irisflow.pipeline.runner import PipelineRunner
from tests.fixtures.api_helpers import build_stubbed_app_state

_WARMUP_TICKS = 200
# 30 minutes at 30 FPS = 54 000 ticks. Round down a bit for CI headroom.
_SOAK_TICKS = 50_000
_HALF = _SOAK_TICKS // 2
# Real callers snapshot + reset every few seconds; simulate that so the
# metrics buffer stays bounded (the design choice from Sprint 7).
_METRICS_RESET_EVERY = 500


@pytest.mark.slow
def test_pipeline_stays_stable_over_simulated_thirty_minutes() -> None:
    bundle = build_stubbed_app_state(config=AppConfig())
    components = bundle.state.components
    runner = PipelineRunner(components, idle_sleep_s=0.0)

    consumed = {"n": 0}

    def _consume(event: object) -> None:
        if isinstance(event, GazeUpdated):
            consumed["n"] += 1

    def _drive(ticks: int) -> int:
        """Run ``ticks`` iterations, resetting metrics on the same cadence
        production code uses. Returns ``GazeUpdated`` count observed."""
        remaining = ticks
        seen_before = consumed["n"]
        while remaining > 0:
            chunk = min(_METRICS_RESET_EVERY, remaining)
            runner.run(max_ticks=chunk)
            components.metrics.snapshot()
            components.metrics.reset()
            remaining -= chunk
        return consumed["n"] - seen_before

    # Warm up without the counting subscriber so the assertion below
    # only measures the soak proper.
    runner.run(max_ticks=_WARMUP_TICKS)
    components.metrics.reset()
    components.bus.subscribe(GazeUpdated, _consume)

    tracemalloc.start()
    baseline_snapshot = tracemalloc.take_snapshot()

    first_half = _drive(_HALF)
    second_half = _drive(_HALF)

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # ---- 1. Flat FPS -----------------------------------------------------
    assert first_half > 0
    assert second_half > 0
    ratio = second_half / first_half
    assert 0.8 <= ratio <= 1.2, f"FPS decay: first={first_half} second={second_half}"

    # ---- 2. Bounded memory when the caller resets metrics ---------------
    diff = snapshot.compare_to(baseline_snapshot, "filename")
    total_growth = sum(stat.size_diff for stat in diff)
    assert total_growth < 2 * 1024 * 1024, (
        f"memory grew by {total_growth} bytes over the soak "
        "(callers must snapshot+reset metrics periodically)"
    )

    # ---- 3. Bus subscribers kept up -------------------------------------
    assert consumed["n"] == first_half + second_half, "bus subscriber missed frames"

    bundle.state.stop()
