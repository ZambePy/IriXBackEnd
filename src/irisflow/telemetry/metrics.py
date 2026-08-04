"""Latency-per-stage and frame-count metrics for the pipeline.

Small on purpose: the recorder is a plain in-memory collector that any
number of stages can time into via :meth:`time`, and a periodic snapshot
(via :meth:`snapshot`) is what downstream sinks report / log. No fancy
histogram engine, no external dependency — Sprint 13 can bolt on
Prometheus / OpenTelemetry once the shape settles.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from irisflow.core.clock import Clock, SystemClock

__all__ = ["MetricsRecorder", "MetricsSnapshot", "StageStats"]


@dataclass(frozen=True, slots=True)
class StageStats:
    """Summary of one stage's latency in a snapshot window."""

    name: str
    count: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    mean_ms: float


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """A point-in-time view of the recorder's state.

    ``duration_s`` is the wall time between the last :meth:`reset` (or
    :meth:`__init__`) and the moment :meth:`snapshot` was taken — the
    denominator for the FPS calculation.
    """

    stages: dict[str, StageStats]
    fps: float
    frames_ok: int
    frames_dropped: int
    frames_face_lost: int
    duration_s: float


@dataclass
class MetricsRecorder:
    """Thread-unsafe recorder — one instance per pipeline runner.

    The pipeline is single-threaded on the consumer side (capture already
    runs in its own thread and hands off through a queue), so adding
    locking here would be pure overhead.
    """

    _clock: Clock = field(default_factory=SystemClock)
    _durations: dict[str, list[float]] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)
    _origin: float = field(init=False)

    def __post_init__(self) -> None:
        self._origin = self._clock.monotonic()

    # ------------------------------------------------------------------ timing
    @contextmanager
    def time(self, stage_name: str) -> Iterator[None]:
        """Time a block against ``stage_name`` — record the duration on exit."""
        start = self._clock.monotonic()
        try:
            yield
        finally:
            duration_ms = (self._clock.monotonic() - start) * 1000.0
            self._durations.setdefault(stage_name, []).append(duration_ms)

    # ------------------------------------------------------------------ counters
    def increment(self, name: str, by: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + by

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> MetricsSnapshot:
        now = self._clock.monotonic()
        duration_s = max(now - self._origin, 1e-9)
        stages: dict[str, StageStats] = {}
        for name, samples in self._durations.items():
            if not samples:
                continue
            arr = np.asarray(samples, dtype=np.float64)
            stages[name] = StageStats(
                name=name,
                count=int(arr.size),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                max_ms=float(arr.max()),
                mean_ms=float(arr.mean()),
            )
        frames_ok = self._counters.get("frames_ok", 0)
        frames_dropped = self._counters.get("frames_dropped", 0)
        frames_face_lost = self._counters.get("frames_face_lost", 0)
        fps = frames_ok / duration_s if duration_s > 0 else 0.0
        return MetricsSnapshot(
            stages=stages,
            fps=fps,
            frames_ok=frames_ok,
            frames_dropped=frames_dropped,
            frames_face_lost=frames_face_lost,
            duration_s=duration_s,
        )

    def reset(self) -> None:
        """Drop all samples/counters and restart the window."""
        self._durations.clear()
        self._counters.clear()
        self._origin = self._clock.monotonic()
