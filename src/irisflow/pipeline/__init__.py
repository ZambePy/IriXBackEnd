"""Pipeline layer — orchestration, state machine, event bus, main loop.

Public surface:

* :class:`EventBus` — the fan-out seam every consumer subscribes to.
* :class:`PipelineStateMachine` — enforces valid state transitions.
* :class:`PipelineComponents` + :func:`build_pipeline` — dependency-
  injected constructor for the collaborators the runner needs.
* :class:`PipelineRunner` — the loop.
"""

from irisflow.pipeline.bus import EventBus, EventHandler
from irisflow.pipeline.orchestrator import PipelineComponents, build_pipeline
from irisflow.pipeline.runner import PipelineRunner, make_stop_handler
from irisflow.pipeline.stages import (
    STAGE_CAPTURE,
    STAGE_DETECTION,
    STAGE_INFERENCE,
    STAGE_PREPROCESS,
)
from irisflow.pipeline.state import PipelineStateMachine

__all__ = [
    "STAGE_CAPTURE",
    "STAGE_DETECTION",
    "STAGE_INFERENCE",
    "STAGE_PREPROCESS",
    "EventBus",
    "EventHandler",
    "PipelineComponents",
    "PipelineRunner",
    "PipelineStateMachine",
    "build_pipeline",
    "make_stop_handler",
]
