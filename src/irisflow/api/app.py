"""FastAPI factory (Sprint 12).

:func:`create_app` builds a fully-wired app: pipeline runner in a
background thread, session hub bridging the bus to WS clients, HTTP
routes for health / config / profiles / calibration, and the
``/ws/gaze`` endpoint.

The factory takes an optional pre-built :class:`AppState` so tests can
inject fakes without paying for a full pipeline stand-up.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from irisflow.api.routes import (
    calibration_router,
    config_router,
    health_router,
    profiles_router,
)
from irisflow.api.session import SessionHub
from irisflow.api.state import AppState
from irisflow.api.ws import router as ws_router
from irisflow.config.loader import load_config
from irisflow.config.schema import AppConfig
from irisflow.control.noop import NoOpCursor
from irisflow.core.interfaces import CursorController
from irisflow.mapping.screen_info import ScreenInfo
from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.safety import RestZone, SafetyGate
from irisflow.pipeline.control_sink import ControlSink
from irisflow.pipeline.orchestrator import PipelineComponents, build_pipeline
from irisflow.pipeline.runner import PipelineRunner

__all__ = ["build_app_state_from_config", "create_app"]


def create_app(
    *,
    state: AppState | None = None,
    config: AppConfig | None = None,
    state_factory: Callable[[AppConfig], AppState] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        state: A pre-wired :class:`AppState` — used by tests to inject
            stub pipelines. When provided the app skips its own build.
        config: Optional config to pass to ``state_factory``; defaults
            to :func:`load_config` when neither ``state`` nor ``config``
            is given.
        state_factory: Custom builder — receives the loaded config and
            returns an :class:`AppState`. Defaults to
            :func:`build_app_state_from_config`.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state is not None:
            resolved = state
        else:
            cfg = config if config is not None else load_config()
            builder = state_factory if state_factory is not None else build_app_state_from_config
            resolved = builder(cfg)
        resolved.start()
        app.state.irisflow = resolved
        try:
            yield
        finally:
            resolved.stop()

    app = FastAPI(
        title="IrisFlow",
        version="0.1.0",
        summary="Eye-tracking backend WebSocket API.",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(config_router)
    app.include_router(profiles_router)
    app.include_router(calibration_router)
    app.include_router(ws_router)
    return app


def build_app_state_from_config(cfg: AppConfig) -> AppState:
    """Build a production :class:`AppState` from a plain :class:`AppConfig`.

    Extracted so tests can call it with a config that swaps out the
    heavy adapters (webcam, real model) via
    :func:`irisflow.pipeline.orchestrator.build_pipeline` overrides.
    """
    screen = ScreenInfo(width_px=1920, height_px=1080)
    cursor: CursorController = NoOpCursor(enabled=False)
    components: PipelineComponents = build_pipeline(
        cfg,
        cursor=cursor,
        screen=screen,
    )
    control_sink = ControlSink(
        bus=components.bus,
        cursor=cursor,
        safety=_build_safety_gate(cfg, components),
        dwell=_build_dwell(cfg),
        cursor_enabled_on_start=False,
        clock=components.clock,
    )
    runner = PipelineRunner(components, watchdog_kick=control_sink.watchdog_kick)
    hub = SessionHub(
        components.bus, screen_width=screen.width_px, screen_height=screen.height_px
    )
    return AppState(
        config=cfg,
        components=components,
        control_sink=control_sink,
        runner=runner,
        hub=hub,
        screen_width=screen.width_px,
        screen_height=screen.height_px,
    )


def _build_safety_gate(cfg: AppConfig, components: PipelineComponents) -> SafetyGate:
    return SafetyGate(
        pause_on_face_lost_ms=cfg.control.safety.pause_on_face_lost_ms,
        rest_zone=RestZone(*cfg.control.safety.rest_zone_px),
        clock=components.clock,
    )


def _build_dwell(cfg: AppConfig) -> DwellClicker:
    return DwellClicker(
        DwellParams(
            radius_px=cfg.control.dwell.radius_px,
            duration_ms=cfg.control.dwell.duration_ms,
            refractory_ms=cfg.control.dwell.refractory_ms,
        )
    )
