"""``/calibration`` HTTP endpoints — status + abort mirror the WS surface.

The WebSocket is the canonical path for calibration (SPRINTS §12 DoD:
"Calibração pode ser conduzida inteiramente pelo WebSocket"). These HTTP
routes exist so operators / smoke tests can inspect state and force a
stop without opening a socket.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from irisflow.api.schemas import CalibrationStatusResponse
from irisflow.api.state import AppState, get_app_state

router = APIRouter(prefix="/calibration", tags=["calibration"])

StateDep = Annotated[AppState, Depends(get_app_state)]


@router.get("/status", response_model=CalibrationStatusResponse)
def status(state: StateDep) -> CalibrationStatusResponse:
    coord = state.calibration
    if coord is None:
        return CalibrationStatusResponse(active=False, phase="idle")
    return CalibrationStatusResponse(
        active=True,
        phase=coord.session.phase.value,
        current_index=coord.session._active_index,
        total=coord.session.total_targets,
        collected=coord.session.current_target_collected,
        profile_id=coord.session.profile_id,
        last_error_px=coord.last_error_px,
    )


@router.post("/abort", status_code=204)
def abort(state: StateDep) -> None:
    coord = state.calibration
    if coord is None:
        raise HTTPException(status_code=409, detail="no calibration in progress")
    state.stop_calibration()
