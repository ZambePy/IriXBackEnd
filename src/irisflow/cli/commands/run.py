"""``irisflow run`` — the fim-a-fim pipeline (Sprint 7).

Prints one line per delivered gaze estimate, plus a periodic metrics
summary. ``--no-cursor`` is honoured trivially: Sprint 7 has no cursor
controller yet, so the flag is accepted for forward compatibility only.

Handles ``SIGINT`` / ``SIGTERM`` cleanly — the DoD says both the camera
and the pipeline threads must be released in under a second.
"""

from __future__ import annotations

import signal
import threading
from pathlib import Path
from typing import Annotated

import typer

from irisflow.config.loader import load_config
from irisflow.core.events import RawGazeReady, StateChanged
from irisflow.core.exceptions import IrisFlowError
from irisflow.pipeline.orchestrator import PipelineComponents, build_pipeline
from irisflow.pipeline.runner import PipelineRunner, make_stop_handler

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
                "Do not move the OS cursor. Sprint 7 default and only "
                "supported mode; kept as a flag for forward compatibility."
            ),
        ),
    ] = True,
    metrics_every: Annotated[
        int,
        typer.Option(
            "--metrics-every",
            min=1,
            help="Print a metrics summary every N delivered gaze frames.",
        ),
    ] = 30,
    quiet_gaze: Annotated[
        bool,
        typer.Option(
            "--quiet-gaze",
            help="Suppress per-frame gaze prints; keep only metrics summaries.",
        ),
    ] = False,
) -> None:
    """Stream normalized gaze coordinates to the terminal until interrupted."""
    del no_cursor  # accepted for forward compat; no cursor sink until Sprint 10
    try:
        cfg = load_config(yaml_path=config) if config is not None else load_config()
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo("Building pipeline...")
    try:
        components = build_pipeline(cfg)
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    runner = PipelineRunner(components)
    _wire_console_sinks(
        components,
        runner=runner,
        metrics_every=metrics_every,
        quiet_gaze=quiet_gaze,
    )

    _install_signal_handlers(runner)
    typer.echo("Pipeline running. Press Ctrl+C to stop.")
    try:
        runner.run()
    finally:
        _print_final_snapshot(components)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
def _wire_console_sinks(
    components: PipelineComponents,
    *,
    runner: PipelineRunner,
    metrics_every: int,
    quiet_gaze: bool,
) -> None:
    counter = {"n": 0}
    lock = threading.Lock()

    def _on_gaze(event: object) -> None:
        if not isinstance(event, RawGazeReady):
            return
        with lock:
            counter["n"] += 1
            n = counter["n"]
        if not quiet_gaze:
            typer.echo(
                f"gaze x={event.x:.3f} y={event.y:.3f} "
                f"conf={event.confidence:.2f} inf_ms={event.inference_ms:.1f}"
            )
        if n % metrics_every == 0:
            _print_snapshot(components)

    def _on_state(event: object) -> None:
        if not isinstance(event, StateChanged):
            return
        typer.echo(f"state: {event.previous.value} -> {event.current.value}")

    components.bus.subscribe(RawGazeReady, _on_gaze)
    components.bus.subscribe(StateChanged, _on_state)
    # Reference `runner` so lint doesn't flag it as unused — the runner is
    # what the SIGINT handler wired later actually stops.
    _ = runner


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


def _print_final_snapshot(components: PipelineComponents) -> None:
    typer.echo("")
    typer.echo("--- final metrics ---")
    _print_snapshot(components)


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
