"""FastAPI + WebSocket surface (Sprint 12).

Public entry point is :func:`create_app`. The rest is implementation
detail — the layer contract prevents anything outside ``api/`` from
importing these modules directly.
"""

from irisflow.api.app import build_app_state_from_config, create_app

__all__ = ["build_app_state_from_config", "create_app"]
