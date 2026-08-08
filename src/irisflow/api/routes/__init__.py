"""HTTP routes for the FastAPI app (Sprint 12)."""

from irisflow.api.routes.calibration import router as calibration_router
from irisflow.api.routes.config import router as config_router
from irisflow.api.routes.health import router as health_router
from irisflow.api.routes.profiles import router as profiles_router

__all__ = [
    "calibration_router",
    "config_router",
    "health_router",
    "profiles_router",
]
