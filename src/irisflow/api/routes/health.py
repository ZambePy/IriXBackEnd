"""``GET /health`` — probe + operator banner."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from irisflow.api.schemas import HealthResponse
from irisflow.api.state import AppState, get_app_state

router = APIRouter(tags=["health"])

StateDep = Annotated[AppState, Depends(get_app_state)]


@router.get("/health", response_model=HealthResponse)
def health(state: StateDep) -> HealthResponse:
    snap = state.components.metrics.snapshot()
    return HealthResponse(
        status="ok" if state.runner_alive else "starting",
        pipeline_state=state.components.state.state.value,
        backend=state.config.model.backend,
        fps=snap.fps,
        frames_ok=snap.frames_ok,
        frames_dropped=snap.frames_dropped,
        frames_face_lost=snap.frames_face_lost,
    )
