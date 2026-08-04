"""Telemetry — metrics, session recording, reports. Sprint 7 introduces
:class:`MetricsRecorder`; Sprint 11 adds :mod:`recorder` and :mod:`report`.
"""

from irisflow.telemetry.metrics import MetricsRecorder, MetricsSnapshot, StageStats

__all__ = ["MetricsRecorder", "MetricsSnapshot", "StageStats"]
