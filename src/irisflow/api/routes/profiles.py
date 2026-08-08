"""``GET /profiles`` — enumerate persisted calibration profiles."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from irisflow.api.schemas import ProfileInfo, ProfileListResponse
from irisflow.api.state import AppState, get_app_state
from irisflow.calibration.store import CalibrationStore
from irisflow.core.exceptions import CalibrationError

router = APIRouter(prefix="/profiles", tags=["profiles"])

StateDep = Annotated[AppState, Depends(get_app_state)]


@router.get("", response_model=ProfileListResponse)
def list_profiles(state: StateDep) -> ProfileListResponse:
    store = CalibrationStore(state.config.calibration.profiles_dir)
    infos: list[ProfileInfo] = []
    for profile_id in store.list_profiles():
        try:
            profile = store.load(profile_id)
        except CalibrationError:
            continue
        infos.append(
            ProfileInfo(
                profile_id=profile.profile_id,
                screen_width=profile.screen_width,
                screen_height=profile.screen_height,
                mean_error_px=profile.mean_error_px,
                p95_error_px=profile.p95_error_px,
                n_samples=profile.n_samples,
                created_at_wall_s=profile.created_at_wall_s,
            )
        )
    return ProfileListResponse(profiles=infos)
