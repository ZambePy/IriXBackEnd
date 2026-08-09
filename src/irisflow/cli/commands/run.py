"""``irisflow run`` — the fim-a-fim pipeline.

Sprint 10 finishes the command:

* wires calibration → mapping → filtering into the runner (Sprint 9
  primitives + Sprint 8 profile persistence);
* honours ``--cursor``: the real :class:`~irisflow.control.cursor.PynputCursor`
  is enabled and the safety layer (kill switch + watchdog + dwell
  refractory) is armed;
* falls back to :class:`~irisflow.control.noop.NoOpCursor` by default so
  the mouse of the developer running the CLI is never touched by
  accident.

Handles ``SIGINT`` / ``SIGTERM`` cleanly — the DoD says both the camera
and the pipeline threads must be released in under a second, and the
mouse must not be left under gaze control if the process crashes.
"""

from __future__ import annotations

import signal
import threading
from pathlib import Path
from typing import Annotated

import typer

from irisflow.config.loader import load_config
from irisflow.config.schema import AppConfig
from irisflow.control.cursor import PynputCursor
from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.kill_switch import KillSwitchListener
from irisflow.control.noop import NoOpCursor
from irisflow.control.safety import RestZone, SafetyGate, Watchdog
from irisflow.core.events import (
    DwellClick,
    GazeUpdated,
    RawGazeReady,
    SafetyPaused,
    SafetyResumed,
    StateChanged,
)
from irisflow.core.exceptions import IrisFlowError
from irisflow.core.interfaces import CursorController
from irisflow.mapping.screen_info import ScreenInfo
from irisflow.pipeline.control_sink import ControlSink
from irisflow.pipeline.orchestrator import PipelineComponents, build_pipeline
from irisflow.pipeline.runner import PipelineRunner, make_stop_handler
from irisflow.telemetry.recorder import SessionRecorder
from irisflow.telemetry.report import (
    build_session_report,
    render_session_report,
)
from irisflow.telemetry.session import SessionHeader

__all__ = ["run"]


def run(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to the YAML config."),
    ] = None,
    no_cursor: Annotated[
        bool,
        typer.Option(
            "--no-cursor",
            help=(
                "Do not move the OS cursor (safe default). Overrides "
                "``control.enabled`` from the config."
            ),
        ),
    ] = True,
    cursor: Annotated[
        bool,
        typer.Option(
            "--cursor",
            help=(
                "Enable OS cursor control via pynput. The kill switch "
                "(default Ctrl+Alt+Esc) releases the mouse."
            ),
        ),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            "-p",
            help="Calibration profile id (loaded from configs/profiles/<id>.json).",
        ),
    ] = None,
    screen_width: Annotated[
        int,
        typer.Option("--screen-width", min=1, help="Target screen width in pixels."),
    ] = 1920,
    screen_height: Annotated[
        int,
        typer.Option("--screen-height", min=1, help="Target screen height in pixels."),
    ] = 1080,
    metrics_every: Annotated[
        int,
        typer.Option(
            "--metrics-every",
            min=1,
            help="Print a metrics summary every N seconds.",
        ),
    ] = 10,
    quiet_gaze: Annotated[
        bool,
        typer.Option(
            "--quiet-gaze",
            help="Suppress per-frame gaze prints; keep only metrics summaries.",
        ),
    ] = False,
    record: Annotated[
        bool,
        typer.Option(
            "--record",
            help=(
                "Record the session to <recordings_dir>/<session_id>.jsonl "
                "for later replay + report."
            ),
        ),
    ] = False,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Explicit session id for the recording; defaults to a timestamp.",
        ),
    ] = None,
) -> None:
    """Stream normalized gaze coordinates to the terminal until interrupted."""
    cursor_enabled = _resolve_cursor_flag(no_cursor=no_cursor, cursor=cursor)

    try:
        cfg = load_config(yaml_path=config) if config is not None else load_config()
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo("Building pipeline...")
    try:
        cursor_controller = _build_cursor(cfg, enabled=cursor_enabled)
        screen = ScreenInfo(width_px=screen_width, height_px=screen_height)
        components = build_pipeline(
            cfg,
            calibration_profile_id=profile,
            cursor=cursor_controller,
            screen=screen,
        )
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    control_sink = _build_control_sink(
        cfg,
        components,
        cursor_controller=cursor_controller,
        cursor_enabled=cursor_enabled,
    )

    recorder: SessionRecorder | None = None
    if record or cfg.telemetry.record_sessions:
        recorder = _build_recorder(
            cfg,
            components,
            profile=profile,
            screen=screen,
            session_id=session_id,
        )

    runner = PipelineRunner(components, watchdog_kick=control_sink.watchdog_kick)
    _wire_console_sinks(
        components,
        runner=runner,
        metrics_every=metrics_every,
        quiet_gaze=quiet_gaze,
        cursor_enabled=cursor_enabled,
    )

    _install_signal_handlers(runner)
    if cursor_enabled:
        typer.echo(
            "Cursor control ENABLED — press "
            f"{cfg.control.safety.kill_switch.upper()} to release the mouse."
        )
    else:
        typer.echo("Cursor control disabled (safe default).")
    typer.echo("Pipeline running. Press Ctrl+C to stop.")

    control_sink.start()
    if recorder is not None:
        recorder.start()
        typer.echo(f"Recording session to {recorder.output_path}")
    try:
        runner.run()
    finally:
        control_sink.stop()
        if recorder is not None:
            recorder.stop()
        _print_final_snapshot(components)
        if recorder is not None:
            _print_final_report(components, recorder)


# ---------------------------------------------------------------------------
# Cursor / control wiring
# ---------------------------------------------------------------------------
def _resolve_cursor_flag(*, no_cursor: bool, cursor: bool) -> bool:
    """--cursor overrides --no-cursor. Default (both defaults) is disabled."""
    if cursor:
        return True
    if no_cursor:
        return False
    return False


def _build_cursor(cfg: AppConfig, *, enabled: bool) -> CursorController:
    if not enabled:
        return NoOpCursor(enabled=False)
    return PynputCursor(rate_limit_hz=cfg.control.rate_limit_hz, enabled=False)


def _build_control_sink(
    cfg: AppConfig,
    components: PipelineComponents,
    *,
    cursor_controller: CursorController,
    cursor_enabled: bool,
) -> ControlSink:
    dwell = DwellClicker(
        DwellParams(
            radius_px=cfg.control.dwell.radius_px,
            duration_ms=cfg.control.dwell.duration_ms,
            refractory_ms=cfg.control.dwell.refractory_ms,
        )
    )
    safety = SafetyGate(
        pause_on_face_lost_ms=cfg.control.safety.pause_on_face_lost_ms,
        rest_zone=RestZone(*cfg.control.safety.rest_zone_px),
        clock=components.clock,
    )
    kill_switch: KillSwitchListener | None = None
    watchdog: Watchdog | None = None
    sink_holder: dict[str, ControlSink | None] = {"sink": None}

    def _on_kill_switch() -> None:
        sink = sink_holder["sink"]
        if sink is not None:
            sink.on_kill_switch()

    def _on_watchdog_stall() -> None:
        sink = sink_holder["sink"]
        if sink is not None:
            sink.on_watchdog_stall()

    if cursor_enabled:
        kill_switch = KillSwitchListener(
            hotkey=cfg.control.safety.kill_switch, on_trigger=_on_kill_switch
        )
        watchdog = Watchdog(
            timeout_ms=cfg.control.safety.watchdog_timeout_ms,
            on_stall=_on_watchdog_stall,
            clock=components.clock,
        )
    sink = ControlSink(
        bus=components.bus,
        cursor=cursor_controller,
        safety=safety,
        dwell=dwell,
        kill_switch=kill_switch,
        watchdog=watchdog,
        cursor_enabled_on_start=cursor_enabled,
        clock=components.clock,
    )
    sink_holder["sink"] = sink
    return sink


# ---------------------------------------------------------------------------
# Console sinks
# ---------------------------------------------------------------------------
def _wire_console_sinks(
    components: PipelineComponents,
    *,
    runner: PipelineRunner,
    metrics_every: int,
    quiet_gaze: bool,
    cursor_enabled: bool,
) -> None:
    lock = threading.Lock()
    import time as _time
    _last_metrics_ts = {"t": _time.monotonic()}

    def _on_raw(event: object) -> None:
        if not isinstance(event, RawGazeReady):
            return
        if not quiet_gaze:
            typer.echo(
                f"raw x={event.x:.3f} y={event.y:.3f} "
                f"conf={event.confidence:.2f} inf_ms={event.inference_ms:.1f}"
            )
        now = _time.monotonic()
        with lock:
            if now - _last_metrics_ts["t"] >= metrics_every:
                _last_metrics_ts["t"] = now
                should_print = True
            else:
                should_print = False
        if should_print:
            _print_snapshot(components)

    def _on_gaze(event: object) -> None:
        if not isinstance(event, GazeUpdated):
            return
        if quiet_gaze:
            return
        typer.echo(
            f"gaze px={event.px} py={event.py} "
            f"fixation={'yes' if event.is_fixation else 'no'} conf={event.confidence:.2f}"
        )

    def _on_state(event: object) -> None:
        if not isinstance(event, StateChanged):
            return
        typer.echo(f"state: {event.previous.value} -> {event.current.value}")

    def _on_click(event: object) -> None:
        if not isinstance(event, DwellClick):
            return
        typer.echo(f"click at ({event.px}, {event.py}) button={event.button}")

    def _on_paused(event: object) -> None:
        if not isinstance(event, SafetyPaused):
            return
        typer.echo(f"[safety] paused reason={event.reason}")

    def _on_resumed(event: object) -> None:
        if not isinstance(event, SafetyResumed):
            return
        typer.echo("[safety] resumed")

    components.bus.subscribe(RawGazeReady, _on_raw)
    components.bus.subscribe(GazeUpdated, _on_gaze)
    components.bus.subscribe(StateChanged, _on_state)
    components.bus.subscribe(DwellClick, _on_click)
    components.bus.subscribe(SafetyPaused, _on_paused)
    components.bus.subscribe(SafetyResumed, _on_resumed)
    # Reference `runner` and `cursor_enabled` so lint doesn't flag them.
    _ = runner
    _ = cursor_enabled


def _print_snapshot(components: PipelineComponents) -> None:
    snap = components.metrics.snapshot()
    typer.echo(
        f"[metrics] fps={snap.fps:.1f} ok={snap.frames_ok} "
        f"dropped={snap.frames_dropped} face_lost={snap.frames_face_lost}"
    )
    for stage, stats in snap.stages.items():
        typer.echo(
            f"  {stage:<11} n={stats.count:>4} "
            f"p50={stats.p50_ms:>6.2f}ms p95={stats.p95_ms:>6.2f}ms "
            f"max={stats.max_ms:>6.2f}ms"
        )
    # If the source tracks dropped frames, show it.
    source = components.source
    if hasattr(source, "dropped_count"):
        typer.echo(f"  capture_dropped (queue): {source.dropped_count}")


def _print_final_snapshot(components: PipelineComponents) -> None:
    typer.echo("")
    typer.echo("--- final metrics ---")
    _print_snapshot(components)


# ---------------------------------------------------------------------------
# Session recording
# ---------------------------------------------------------------------------
def _build_recorder(
    cfg: AppConfig,
    components: PipelineComponents,
    *,
    profile: str | None,
    screen: ScreenInfo,
    session_id: str | None,
) -> SessionRecorder:
    import datetime as _dt

    resolved_id = session_id or _dt.datetime.now(_dt.UTC).strftime("session-%Y%m%dT%H%M%S")
    output_path = cfg.telemetry.recordings_dir / f"{resolved_id}.jsonl"
    header = SessionHeader(
        session_id=resolved_id,
        wall_start_s=components.clock.wall(),
        screen_width_px=screen.width_px,
        screen_height_px=screen.height_px,
        calibration_profile_id=profile,
        config_snapshot=cfg.model_dump(mode="json"),
    )
    return SessionRecorder(
        output_path=output_path,
        header=header,
        subscribe=components.bus.subscribe,
        unsubscribe=components.bus.unsubscribe,
        clock=components.clock,
    )


def _print_final_report(components: PipelineComponents, recorder: SessionRecorder) -> None:
    from irisflow.telemetry.session import read_session_file

    try:
        header, events = read_session_file(recorder.output_path)
    except Exception as exc:
        typer.echo(f"[warn] could not read back recording for report: {exc}")
        return
    report = build_session_report(
        session_id=header.session_id,
        events=events,
        metrics=components.metrics.snapshot(),
    )
    typer.echo("")
    typer.echo("--- session report ---")
    typer.echo(render_session_report(report))


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def _install_signal_handlers(runner: PipelineRunner) -> None:
    handler = make_stop_handler(runner)
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Signal handlers can only be installed on the main thread and
            # SIGTERM is unavailable on Windows console apps in some cases.
            continue
