"""The pipeline loop — captures a frame, walks it through every stage,
publishes the result, and repeats until asked to stop.

Design invariants (SPRINTS.md §7 DoD):

* **Never dies on a stage exception.** Any error caught inside :meth:`_tick`
  is logged, counted as a drop, and the loop continues on the next frame.
* **Clean shutdown < 1 s.** The runner obeys a :class:`threading.Event`
  stop flag; the CLI wires it to ``SIGINT``.
* **No hardware in tests.** Every collaborator is behind a Protocol —
  swap in synthetic sources / stubbed detectors and the loop runs
  offline in CI.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable

from irisflow.core.events import (
    FaceAcquired,
    FaceLost,
    PipelineState,
    RawGazeReady,
)
from irisflow.logging import bind, clear_context, get_logger
from irisflow.pipeline.orchestrator import PipelineComponents
from irisflow.pipeline.stages import (
    STAGE_CAPTURE,
    STAGE_DETECTION,
    STAGE_INFERENCE,
    STAGE_PREPROCESS,
)

__all__ = ["PipelineRunner"]


_DEFAULT_IDLE_SLEEP_S = 0.005


class PipelineRunner:
    """Own the main loop lifecycle for one :class:`PipelineComponents`.

    Args:
        components: The pre-wired stage collaborators (built by
            :func:`~irisflow.pipeline.orchestrator.build_pipeline`).
        idle_sleep_s: How long to sleep when the source returns no frame.
            Small enough to keep responsiveness high, big enough to not
            burn a CPU core doing nothing.
    """

    __slots__ = (
        "_components",
        "_face_last_seen",
        "_face_lost_streak",
        "_idle_sleep_s",
        "_last_face_present",
        "_log",
        "_stop",
    )

    def __init__(
        self,
        components: PipelineComponents,
        *,
        idle_sleep_s: float = _DEFAULT_IDLE_SLEEP_S,
    ) -> None:
        self._components = components
        self._idle_sleep_s = idle_sleep_s
        self._stop = threading.Event()
        self._log = get_logger("irisflow.pipeline.runner")
        self._last_face_present = False
        self._face_last_seen: float | None = None
        self._face_lost_streak = 0

    # ------------------------------------------------------------------ API
    @property
    def is_running(self) -> bool:
        return not self._stop.is_set()

    def stop(self) -> None:
        """Signal the loop to exit at the next tick boundary."""
        self._stop.set()

    def run(self, *, max_ticks: int | None = None) -> None:
        """Run until :meth:`stop` is called or ``max_ticks`` is reached.

        ``max_ticks`` is what integration tests use to bound the loop;
        production callers leave it as ``None``.
        """
        components = self._components
        components.source.open()
        components.state.transition_to(PipelineState.TRACKING)
        try:
            ticks = 0
            while not self._stop.is_set():
                if max_ticks is not None and ticks >= max_ticks:
                    break
                self._tick()
                ticks += 1
        finally:
            self._shutdown()

    # ------------------------------------------------------------------ loop body
    def _tick(self) -> None:
        components = self._components
        metrics = components.metrics
        with metrics.time(STAGE_CAPTURE):
            frame = components.source.read()
        if frame is None:
            components.clock.sleep(self._idle_sleep_s)
            return

        bind(frame_id=frame.frame_id)
        try:
            with metrics.time(STAGE_DETECTION):
                detection = components.detector.detect(frame)
            if detection is None:
                self._handle_face_lost(frame.frame_id, frame.timestamp)
                metrics.increment("frames_face_lost")
                return
            self._handle_face_acquired(frame.frame_id, frame.timestamp)

            with metrics.time(STAGE_PREPROCESS):
                model_input = components.preprocessor.build(frame, detection)
            with metrics.time(STAGE_INFERENCE):
                raw = components.estimator.predict(model_input)

            components.bus.publish(
                RawGazeReady(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    x=raw.x,
                    y=raw.y,
                    confidence=raw.confidence,
                    inference_ms=raw.inference_ms,
                )
            )
            metrics.increment("frames_ok")
        except Exception as exc:
            self._log.warning(
                "runner.tick_failed", frame_id=frame.frame_id, error=str(exc)
            )
            metrics.increment("frames_dropped")
        finally:
            clear_context()

    # ------------------------------------------------------------------ state transitions
    def _handle_face_lost(self, frame_id: int, timestamp: float) -> None:
        self._face_lost_streak += 1
        if not self._last_face_present:
            # Already in LOST — just count.
            return
        self._last_face_present = False
        self._components.bus.publish(
            FaceLost(frame_id=frame_id, timestamp=timestamp, duration_ms=0.0)
        )
        # Transition may be rejected if we're already IDLE mid-shutdown — swallow.
        with contextlib.suppress(ValueError):
            self._components.state.transition_to(PipelineState.LOST)

    def _handle_face_acquired(self, frame_id: int, timestamp: float) -> None:
        self._face_lost_streak = 0
        if self._last_face_present:
            return
        self._last_face_present = True
        if self._components.state.state == PipelineState.LOST:
            with contextlib.suppress(ValueError):
                self._components.state.transition_to(PipelineState.TRACKING)
        self._components.bus.publish(
            FaceAcquired(frame_id=frame_id, timestamp=timestamp)
        )

    # ------------------------------------------------------------------ shutdown
    def _shutdown(self) -> None:
        components = self._components
        try:
            components.source.close()
        except Exception as exc:
            self._log.warning("runner.source_close_failed", error=str(exc))
        if components.state.state != PipelineState.IDLE:
            with contextlib.suppress(ValueError):
                components.state.transition_to(PipelineState.IDLE)


def make_stop_handler(runner: PipelineRunner) -> Callable[..., None]:
    """Return a ``signal.signal``-compatible callback that stops ``runner``.

    Handy for the CLI so it doesn't have to know about the runner's
    internals: ``signal.signal(SIGINT, make_stop_handler(runner))``.
    """

    def _handler(*_: object) -> None:
        runner.stop()

    return _handler
