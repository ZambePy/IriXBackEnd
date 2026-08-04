"""Named stage constants.

Central definition of the stage-name strings the runner uses when timing
into :class:`~irisflow.telemetry.metrics.MetricsRecorder`. Keeping the
strings in one place (rather than sprinkled through ``runner.py``) means
metrics dashboards and log filters can trust a stable vocabulary.
"""

from __future__ import annotations

from typing import Final

__all__ = ["STAGE_CAPTURE", "STAGE_DETECTION", "STAGE_INFERENCE", "STAGE_PREPROCESS"]

STAGE_CAPTURE: Final[str] = "capture"
STAGE_DETECTION: Final[str] = "detection"
STAGE_PREPROCESS: Final[str] = "preprocess"
STAGE_INFERENCE: Final[str] = "inference"
