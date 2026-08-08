"""``GET/PATCH /config`` — read + narrow-scope runtime tweaks."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from irisflow.api.schemas import ConfigResponse
from irisflow.api.state import AppState, get_app_state

router = APIRouter(prefix="/config", tags=["config"])

StateDep = Annotated[AppState, Depends(get_app_state)]


# The frontend may only tweak parameters that are safe to change mid-run
# — anything that would require rebuilding the pipeline is refused.
_MUTABLE_PATHS: set[str] = {
    "control.dwell.radius_px",
    "control.dwell.duration_ms",
    "control.dwell.refractory_ms",
    "control.enabled",
    "filtering.one_euro.min_cutoff",
    "filtering.one_euro.beta",
    "filtering.one_euro.d_cutoff",
    "filtering.fixation.velocity_threshold_px_per_s",
}


@router.get("", response_model=ConfigResponse)
def get_config(state: StateDep) -> ConfigResponse:
    return ConfigResponse(config=state.config.model_dump(mode="json"))


@router.patch("", response_model=ConfigResponse)
def patch_config(updates: dict[str, Any], state: StateDep) -> ConfigResponse:
    """Apply a shallow patch — dotted keys under :data:`_MUTABLE_PATHS`.

    Anything else is refused with 400 so the frontend can't silently
    ask for a mid-run camera resolution change (which would require a
    full pipeline rebuild).
    """
    if not isinstance(updates, dict) or not updates:
        raise HTTPException(status_code=400, detail="patch body must be a non-empty object")
    for key in updates:
        if key not in _MUTABLE_PATHS:
            raise HTTPException(
                status_code=400,
                detail=f"field not mutable at runtime: {key!r}",
            )
    try:
        state.apply_runtime_patch(updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConfigResponse(config=state.config.model_dump(mode="json"))
