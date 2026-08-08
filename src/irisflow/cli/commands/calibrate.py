"""``irisflow calibrate --profile <name>`` — walk a user through a session.

Reuses the Sprint 7 pipeline (capture → detection → preprocess →
inference) to source :class:`~irisflow.core.types.RawGaze` samples, then
runs them through a :class:`~irisflow.calibration.session.CalibrationSession`
that renders each target as a console prompt (``[calibrate] look at
target 3/9 at nx=0.500 ny=0.500...``). No GUI here — the WebSocket
frontend in Sprint 12 will drive a graphical version by publishing the
same :class:`CalibrationProgress` events on the bus.

Because Sprint 8 has no persistence for calibration in the pipeline yet
(the runner ignores it), the fitted profile is only saved to disk — it
is not applied to the running pipeline. That wiring lands in Sprint 9
alongside the mapping/filter stages.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Annotated

import typer

from irisflow.calibration.session import CalibrationSession
from irisflow.calibration.store import CalibrationStore
from irisflow.config.loader import load_config
from irisflow.core.events import CalibrationProgress, RawGazeReady
from irisflow.core.exceptions import IrisFlowError
from irisflow.core.types import RawGaze
from irisflow.pipeline.orchestrator import build_pipeline
from irisflow.pipeline.runner import PipelineRunner

__all__ = ["calibrate"]


def calibrate(
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            "-p",
            help="Profile id. The fitted model is saved as <profile>.json.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to the YAML config."),
    ] = None,
    screen_width: Annotated[
        int,
        typer.Option(
            "--screen-width",
            min=1,
            help="Target screen width in pixels (used only for quality metrics).",
        ),
    ] = 1920,
    screen_height: Annotated[
        int,
        typer.Option(
            "--screen-height",
            min=1,
            help="Target screen height in pixels (used only for quality metrics).",
        ),
    ] = 1080,
    prompt_seconds: Annotated[
        float,
        typer.Option(
            "--prompt-seconds",
            min=0.1,
            help="Wall-clock time budget per target before we move on.",
        ),
    ] = 4.0,
) -> None:
    """Run a full calibration session and persist the fitted profile."""
    try:
        cfg = load_config(yaml_path=config) if config is not None else load_config()
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Building pipeline for calibration '{profile}'...")
    try:
        components = build_pipeline(cfg)
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    session = CalibrationSession(
        profile_id=profile,
        screen_width=screen_width,
        screen_height=screen_height,
        point_count=cfg.calibration.points,
        samples_per_point=cfg.calibration.samples_per_point,
        stabilization_ms=cfg.calibration.stabilization_ms,
        model_kind=cfg.calibration.model_kind,
        max_residual_px=cfg.calibration.max_residual_px,
        polynomial_degree=cfg.calibration.polynomial_degree,
        publisher=components.bus.publish,
        clock=components.clock,
    )
    _wire_progress_echo(components.bus.subscribe, echo=typer.echo)

    latest_sample = _LatestSampleSink()
    components.bus.subscribe(RawGazeReady, latest_sample.on_event)

    runner = PipelineRunner(components)
    background = threading.Thread(target=runner.run, daemon=True)
    background.start()
    try:
        session.start()
        _drive_targets(
            session=session,
            latest_sample=latest_sample,
            prompt_seconds=prompt_seconds,
        )
        outcome = session.finalize()
    except IrisFlowError as exc:
        runner.stop()
        typer.echo(f"[error] calibration failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    finally:
        runner.stop()
        background.join(timeout=2.0)

    store = CalibrationStore(cfg.calibration.profiles_dir)
    saved_at = store.save(outcome.profile)
    typer.echo("")
    typer.echo(
        f"[calibrate] fit mean_error_px={outcome.report.mean_error_px:.1f} "
        f"p95={outcome.report.p95_error_px:.1f} max={outcome.report.max_error_px:.1f}"
    )
    typer.echo(f"[calibrate] hold-out mean_error_px={outcome.holdout_report.mean_error_px:.1f}")
    if outcome.accepted:
        typer.echo(f"[calibrate] accepted; profile saved to {saved_at}")
    else:
        typer.echo(
            f"[calibrate] REJECTED (mean {outcome.holdout_report.mean_error_px:.1f}px > "
            f"threshold {cfg.calibration.max_residual_px:.1f}px). Bad targets: "
            f"{outcome.bad_targets}. Profile still saved to {saved_at}."
        )


# ---------------------------------------------------------------------------
# Sinks / helpers
# ---------------------------------------------------------------------------
class _LatestSampleSink:
    """Thread-safe holder for the most recent :class:`RawGazeReady`."""

    __slots__ = ("_event", "_lock")

    def __init__(self) -> None:
        self._event: RawGazeReady | None = None
        self._lock = threading.Lock()

    def on_event(self, event: object) -> None:
        if not isinstance(event, RawGazeReady):
            return
        with self._lock:
            self._event = event

    def pop(self) -> RawGazeReady | None:
        with self._lock:
            event = self._event
            self._event = None
        return event


def _drive_targets(
    *,
    session: CalibrationSession,
    latest_sample: _LatestSampleSink,
    prompt_seconds: float,
) -> None:
    """Iterate through every target, feeding samples until each completes."""
    for target in session.targets:
        session.begin_target(target.index)
        deadline = time.monotonic() + prompt_seconds
        while not session.is_current_target_done and time.monotonic() < deadline:
            sample = latest_sample.pop()
            if sample is None:
                time.sleep(0.010)
                continue
            raw = RawGaze(
                x=sample.x,
                y=sample.y,
                confidence=sample.confidence,
                inference_ms=sample.inference_ms,
            )
            session.push_sample(raw, sample.timestamp)
        if not session.is_current_target_done:
            if session.current_target_collected > 0:
                session.force_complete_active()
            else:
                session.abort_target(target.index)


def _wire_progress_echo(subscribe: object, *, echo: object) -> None:
    def _on_progress(event: object) -> None:
        if not isinstance(event, CalibrationProgress):
            return
        echo(  # type: ignore[operator]
            f"[calibrate] target {event.index + 1}/{event.total} "
            f"nx={event.target_x:.2f} ny={event.target_y:.2f} "
            f"phase={event.phase}"
        )

    subscribe(CalibrationProgress, _on_progress)  # type: ignore[operator]
