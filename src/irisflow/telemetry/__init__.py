"""Telemetry — metrics, session recording, replay, reports.

Sprint 7 introduced :class:`MetricsRecorder`. Sprint 11 adds session
recording (:class:`SessionRecorder`), deterministic replay
(:class:`SessionReplayer`) and reports (:class:`SessionReport`).
"""

from irisflow.telemetry.metrics import MetricsRecorder, MetricsSnapshot, StageStats
from irisflow.telemetry.recorder import SessionRecorder
from irisflow.telemetry.report import (
    SessionReport,
    build_session_report,
    render_session_report,
)
from irisflow.telemetry.session import (
    SCHEMA_VERSION,
    SessionHeader,
    SessionRecordError,
    event_from_line,
    event_to_line,
    header_from_line,
    header_to_line,
    iter_session_events,
    read_session_file,
    write_session_file,
)

__all__ = [
    "SCHEMA_VERSION",
    "MetricsRecorder",
    "MetricsSnapshot",
    "SessionHeader",
    "SessionRecordError",
    "SessionRecorder",
    "SessionReport",
    "StageStats",
    "build_session_report",
    "event_from_line",
    "event_to_line",
    "header_from_line",
    "header_to_line",
    "iter_session_events",
    "read_session_file",
    "render_session_report",
    "write_session_file",
]
