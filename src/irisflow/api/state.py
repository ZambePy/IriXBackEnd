"""In-process state shared between HTTP routes and the WS handler.

Everything the API surface needs — the config, the wired pipeline
components, the runner thread, the session hub, the optional calibration
coordinator — lives on one :class:`AppState`. FastAPI dependencies fetch
it via :func:`get_app_state` (populated from ``request.app.state``).

The AppState never runs tracking logic itself. It owns the *lifetime*
of the pipeline and provides thin helpers the routes call — the API
layer contract from SPRINTS §11.3 rule 4 ("API translates, doesn't
decide").
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request

from irisflow.api.schemas import CalibrationResultMessage
from irisflow.calibration.session import CalibrationSession, SessionPhase
from irisflow.calibration.store import CalibrationStore
from irisflow.config.schema import AppConfig
from irisflow.core.events import (
    Event,
    PipelineState,
    RawGazeReady,
)
from irisflow.core.exceptions import CalibrationError, IrisFlowError
from irisflow.core.types import RawGaze
from irisflow.pipeline.control_sink import ControlSink
from irisflow.pipeline.orchestrator import PipelineComponents
from irisflow.pipeline.runner import PipelineRunner

if TYPE_CHECKING:
    from irisflow.api.session import SessionHub

__all__ = [
    "AppState",
    "CalibrationCoordinator",
    "get_app_state",
]


def _current_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


@dataclass
class CalibrationCoordinator:
    """Bridges the WS calibration protocol with :class:`CalibrationSession`.

    The API layer instantiates one per active calibration and tears it
    down on completion or abort. Raw gaze samples are pulled off the bus
    and fed to :meth:`CalibrationSession.push_sample`. When the last
    target is collected we finalize, persist the profile, and broadcast
    a :class:`CalibrationResultMessage`.
    """

    session: CalibrationSession
    profiles_dir: Path
    bus_subscribe: Callable[[type, Callable[[Event], None]], None]
    bus_unsubscribe: Callable[[type, Callable[[Event], None]], None]
    on_result: Callable[[CalibrationResultMessage], None]
    max_residual_px: float
    _started: bool = field(default=False, init=False)
    last_error_px: float | None = field(default=None, init=False)

    def start(self) -> None:
        if self._started:
            return
        self.session.start()
        self.bus_subscribe(RawGazeReady, self._on_raw)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.bus_unsubscribe(RawGazeReady, self._on_raw)
        self._started = False

    def request_target(self, index: int) -> None:
        """Client displayed target ``index`` and is ready to collect."""
        if not self._started:
            raise CalibrationError("calibration not started")
        self.session.begin_target(index)

    def _on_raw(self, event: Event) -> None:
        if not isinstance(event, RawGazeReady):
            return
        if self.session.phase != SessionPhase.COLLECTING:
            return
        if self.session.active_target is None:
            return
        raw = RawGaze(
            x=event.x,
            y=event.y,
            confidence=event.confidence,
            inference_ms=event.inference_ms,
        )
        try:
            self.session.push_sample(raw, event.timestamp)
        except CalibrationError:
            return
        # Auto-finalize once every target has been collected. The FE is
        # still in the loop for target *transitions* (it must send a new
        # ``calibration_ready`` between targets), but once the last one
        # settles we don't need another round-trip.
        if self.session.is_collection_complete:
            self._finalize()

    def _finalize(self) -> None:
        from irisflow.calibration.store import CalibrationStore as _Store

        try:
            outcome = self.session.finalize()
        except CalibrationError as exc:
            self.on_result(
                CalibrationResultMessage(
                    ts=0.0,
                    profile_id=self.session.profile_id,
                    error_px=float("nan"),
                    worst_points=[],
                    accepted=False,
                )
            )
            self.stop()
            raise exc
        # Persist the profile so /profiles picks it up.
        store = _Store(self.profiles_dir)
        store.save(outcome.profile)
        self.last_error_px = outcome.holdout_report.mean_error_px
        self.on_result(
            CalibrationResultMessage(
                ts=0.0,
                profile_id=outcome.profile.profile_id,
                error_px=outcome.holdout_report.mean_error_px,
                worst_points=list(outcome.bad_targets),
                accepted=outcome.accepted,
            )
        )
        self.stop()


@dataclass
class AppState:
    """Everything the API needs, wired once by :func:`create_app`.

    ``runner_thread`` runs the pipeline; the FastAPI app runs on the
    asyncio event loop. The two communicate only via
    :class:`~irisflow.pipeline.bus.EventBus` (thread → asyncio via
    :class:`~irisflow.api.session.SessionHub`) and the runner-owned stop
    :class:`threading.Event`.
    """

    config: AppConfig
    components: PipelineComponents
    control_sink: ControlSink
    runner: PipelineRunner
    hub: SessionHub
    screen_width: int
    screen_height: int
    runner_thread: threading.Thread | None = None
    calibration: CalibrationCoordinator | None = None
    _stop_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Bring up the pipeline in a background thread."""
        self.hub.start()
        self.control_sink.start()
        thread = threading.Thread(target=self._run_pipeline, name="irisflow-pipeline", daemon=True)
        self.runner_thread = thread
        thread.start()

    def _run_pipeline(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover — defensive net
            self.runner.run()

    def stop(self) -> None:
        """Signal the pipeline to shut down and wait for it (bounded)."""
        with self._stop_lock:
            self.runner.stop()
            if self.runner_thread is not None:
                self.runner_thread.join(timeout=2.0)
                self.runner_thread = None
            self.control_sink.stop()
            self.hub.stop()

    @property
    def runner_alive(self) -> bool:
        return self.runner_thread is not None and self.runner_thread.is_alive()

    # ------------------------------------------------------------------ runtime patch
    def apply_runtime_patch(self, updates: dict[str, Any]) -> None:
        """Apply narrow-scope tweaks that don't require rebuilding the pipeline.

        Only the fields in
        :data:`~irisflow.api.routes.config._MUTABLE_PATHS` may reach this
        method; the route already validated the key set.
        """
        current = self.config.model_dump()
        for dotted, value in updates.items():
            _set_dotted(current, dotted, value)
        # Re-validate through Pydantic — bad values raise ValidationError.
        try:
            new_cfg = AppConfig.model_validate(current)
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        self.config = new_cfg
        # Rewire the pieces that consume the mutated fields.
        if any(k.startswith("control.dwell.") for k in updates):
            from irisflow.control.dwell import DwellParams

            self.control_sink._dwell._params = DwellParams(
                radius_px=new_cfg.control.dwell.radius_px,
                duration_ms=new_cfg.control.dwell.duration_ms,
                refractory_ms=new_cfg.control.dwell.refractory_ms,
            )

    # ------------------------------------------------------------------ calibration
    def start_calibration(
        self,
        *,
        profile_id: str,
        points: int,
        screen_width: int,
        screen_height: int,
        publisher: Callable[[Event], None],
        broadcast_result: Callable[[CalibrationResultMessage], None],
    ) -> CalibrationCoordinator:
        if self.calibration is not None:
            raise CalibrationError("calibration already in progress")
        # Ensure LOST / IDLE isn't blocking transitions when the client
        # asks for calibration; pipeline stays in TRACKING throughout.
        session = CalibrationSession(
            profile_id=profile_id,
            screen_width=screen_width,
            screen_height=screen_height,
            point_count=points,  # type: ignore[arg-type]
            samples_per_point=self.config.calibration.samples_per_point,
            stabilization_ms=self.config.calibration.stabilization_ms,
            model_kind=self.config.calibration.model_kind,
            max_residual_px=self.config.calibration.max_residual_px,
            polynomial_degree=self.config.calibration.polynomial_degree,
            publisher=publisher,
            clock=self.components.clock,
        )
        coord = CalibrationCoordinator(
            session=session,
            profiles_dir=self.config.calibration.profiles_dir,
            bus_subscribe=self.components.bus.subscribe,
            bus_unsubscribe=self.components.bus.unsubscribe,
            on_result=broadcast_result,
            max_residual_px=self.config.calibration.max_residual_px,
        )
        coord.start()
        self.calibration = coord
        return coord

    def stop_calibration(self) -> None:
        if self.calibration is None:
            return
        self.calibration.stop()
        self.calibration = None

    # ------------------------------------------------------------------ profile hot-swap
    def set_active_profile(self, profile_id: str) -> None:
        """Load a calibration profile from disk and swap it in for the runner."""
        store = CalibrationStore(self.config.calibration.profiles_dir)
        try:
            profile = store.load(profile_id)
        except CalibrationError as exc:
            raise IrisFlowError(str(exc)) from exc
        self.components.calibration_model = profile.model

    # ------------------------------------------------------------------ pipeline controls
    def request_pipeline_pause(self) -> None:
        try:
            self.components.state.transition_to(PipelineState.PAUSED)
        except ValueError:
            return

    def request_pipeline_resume(self) -> None:
        try:
            self.components.state.transition_to(PipelineState.TRACKING)
        except ValueError:
            return


def _set_dotted(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = target
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency — pulls the state out of ``app.state``."""
    state: AppState | None = getattr(request.app.state, "irisflow", None)
    if state is None:
        raise HTTPException(status_code=503, detail="pipeline not initialized")
    return state
