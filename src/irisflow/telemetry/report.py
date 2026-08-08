"""Session report generation (Sprint 11).

Reads a session file and produces a structured :class:`SessionReport`
with the numbers the CLI prints at the end of every ``irisflow run``
and every ``irisflow replay``. The report is intentionally cheap to
compute — no pandas, no matplotlib — so it can run inline at shutdown
without any noticeable delay.

Metrics computed:

* **Duration**: wall-clock span between the first and last event.
* **FPS**: ``len(GazeUpdated) / duration``.
* **% rosto perdido**: fraction of the run spent in the ``LOST`` state
  (approximated from ``StateChanged`` + tick timestamps).
* **Jitter RMS**: root mean square of successive pixel deltas of
  :class:`GazeUpdated` samples that carry ``is_fixation == True``.
  Matches the metric the Sprint 9 DoD used.
* **Clicks**: total :class:`DwellClick` count.
* **Latency**: optional — comes from the runtime
  :class:`~irisflow.telemetry.metrics.MetricsSnapshot`. Replay does not
  reproduce timings, so the CLI passes the live snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from irisflow.core.events import (
    DwellClick,
    Event,
    FaceAcquired,
    FaceLost,
    GazeUpdated,
    PipelineState,
    StateChanged,
)
from irisflow.telemetry.metrics import MetricsSnapshot, StageStats

__all__ = ["SessionReport", "build_session_report", "render_session_report"]


@dataclass(frozen=True, slots=True)
class SessionReport:
    """One-shot report of a session — safe to log, pickle, or serialise."""

    session_id: str
    duration_s: float
    frames_ok: int
    fps: float
    face_lost_fraction: float
    face_lost_events: int
    face_acquired_events: int
    jitter_px_rms: float
    click_count: int
    stage_latencies_ms: dict[str, StageStats] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_session_report(
    *,
    session_id: str,
    events: Iterable[Event],
    metrics: MetricsSnapshot | None = None,
) -> SessionReport:
    """Compute a :class:`SessionReport` from a stream of events + snapshot.

    Args:
        session_id: The id from :class:`~irisflow.telemetry.session.SessionHeader`.
        events: Ordered iterable of every event in the session.
        metrics: Runtime latency snapshot. May be ``None`` (replay path).
    """
    events_list = list(events)
    gaze_events = [e for e in events_list if isinstance(e, GazeUpdated)]
    click_events = [e for e in events_list if isinstance(e, DwellClick)]
    face_lost_events = [e for e in events_list if isinstance(e, FaceLost)]
    face_acquired_events = [e for e in events_list if isinstance(e, FaceAcquired)]
    state_events = [e for e in events_list if isinstance(e, StateChanged)]

    duration_s = _duration_from_events(events_list)
    fps = (len(gaze_events) / duration_s) if duration_s > 0 else 0.0
    jitter = _jitter_rms(gaze_events)
    face_lost_fraction = _lost_fraction(state_events, duration_s)

    stage_stats: dict[str, StageStats] = {}
    if metrics is not None:
        stage_stats = dict(metrics.stages)

    return SessionReport(
        session_id=session_id,
        duration_s=duration_s,
        frames_ok=len(gaze_events),
        fps=fps,
        face_lost_fraction=face_lost_fraction,
        face_lost_events=len(face_lost_events),
        face_acquired_events=len(face_acquired_events),
        jitter_px_rms=jitter,
        click_count=len(click_events),
        stage_latencies_ms=stage_stats,
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_session_report(report: SessionReport) -> str:
    """Human-friendly multi-line rendering. Used by the CLI."""
    lines = [
        f"session       {report.session_id}",
        f"duration      {report.duration_s:.2f} s",
        f"frames_ok     {report.frames_ok}",
        f"fps           {report.fps:.2f}",
        f"face_lost     {report.face_lost_fraction * 100.0:.1f}% "
        f"({report.face_lost_events} events, "
        f"{report.face_acquired_events} recoveries)",
        f"jitter_rms_px {report.jitter_px_rms:.2f}",
        f"clicks        {report.click_count}",
    ]
    if report.stage_latencies_ms:
        lines.append("stage latencies (ms):")
        for name, stats in report.stage_latencies_ms.items():
            lines.append(
                f"  {name:<11} n={stats.count:>4} "
                f"p50={stats.p50_ms:>6.2f} p95={stats.p95_ms:>6.2f} "
                f"max={stats.max_ms:>6.2f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _duration_from_events(events: Sequence[Event]) -> float:
    """First → last recorded timestamp. Ignores events without a ts field."""
    stamps: list[float] = []
    for e in events:
        ts = getattr(e, "timestamp", None)
        if isinstance(ts, (int, float)):
            stamps.append(float(ts))
    if not stamps:
        return 0.0
    return max(stamps) - min(stamps)


def _jitter_rms(gaze_events: Sequence[GazeUpdated]) -> float:
    """Root mean square of consecutive-sample px deltas during fixation.

    The Sprint 9 target was ≤ 15 px RMS at rest. The report computes
    only over consecutive frames both flagged ``is_fixation`` so a
    saccade doesn't inflate the number.
    """
    deltas: list[float] = []
    prev: GazeUpdated | None = None
    for sample in gaze_events:
        if prev is not None and prev.is_fixation and sample.is_fixation:
            dx = sample.px - prev.px
            dy = sample.py - prev.py
            deltas.append(math.hypot(dx, dy))
        prev = sample
    if not deltas:
        return 0.0
    mean_sq = sum(d * d for d in deltas) / len(deltas)
    return math.sqrt(mean_sq)


def _lost_fraction(state_events: Sequence[StateChanged], duration_s: float) -> float:
    """Fraction of ``duration_s`` spent in :class:`PipelineState.LOST`.

    Reconstructs occupancy from consecutive :class:`StateChanged` pairs
    plus an implicit final "still in this state until the end" segment.
    """
    if duration_s <= 0.0 or not state_events:
        return 0.0
    sorted_events = sorted(state_events, key=lambda e: e.timestamp)
    lost_time = 0.0
    origin = sorted_events[0].timestamp
    end = sorted_events[-1].timestamp
    for prev, curr in pairwise(sorted_events):
        if prev.current == PipelineState.LOST:
            lost_time += max(0.0, curr.timestamp - prev.timestamp)
    if sorted_events[-1].current == PipelineState.LOST:
        # No successor tick — assume the LOST state persisted until the
        # last recorded timestamp anywhere in the file.
        lost_time += max(0.0, duration_s - (end - origin))
    fraction = lost_time / duration_s
    return max(0.0, min(1.0, fraction))
