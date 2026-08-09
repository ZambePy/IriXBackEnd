"""Named stage constants.

Central definition of the stage-name strings the runner uses when timing
into :class:`~irisflow.telemetry.metrics.MetricsRecorder`. Keeping the
strings in one place (rather than sprinkled through ``runner.py``) means
metrics dashboards and log filters can trust a stable vocabulary.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "STAGE_CALIBRATE",
    "STAGE_CAPTURE",
    "STAGE_CONTROL",
    "STAGE_DETECTION",
    "STAGE_FILTER",
    "STAGE_INFERENCE",
    "STAGE_MAP",
    "STAGE_PREPROCESS",
    "STAGE_TICK_WALL",
]

STAGE_CAPTURE: Final[str] = "capture"
STAGE_DETECTION: Final[str] = "detection"
STAGE_PREPROCESS: Final[str] = "preprocess"
STAGE_INFERENCE: Final[str] = "inference"
STAGE_CALIBRATE: Final[str] = "calibrate"
STAGE_MAP: Final[str] = "map"
STAGE_FILTER: Final[str] = "filter"
STAGE_CONTROL: Final[str] = "control"
STAGE_TICK_WALL: Final[str] = "tick_wall"
