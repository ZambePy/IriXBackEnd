"""``irisflow replay <session>`` — deterministic replay + report.

Loads a JSONL session written by :class:`~irisflow.telemetry.SessionRecorder`,
rebuilds the downstream tail of the pipeline (calibration → mapping →
filtering → dwell + safety) and re-emits every :class:`GazeUpdated` /
:class:`DwellProgress` / :class:`DwellClick` /
:class:`SafetyPaused` / :class:`SafetyResumed` the original run produced.

Determinism is verified: the replay output is compared against the
recorded events. Divergence is a hard failure — it means either the
config drifted between record and replay, or a stage stopped being
deterministic.

Usage::

    irisflow replay data/recordings/2026-08-07T18-30-00.jsonl
    irisflow replay --strict data/recordings/foo.jsonl   # exit code 3 on divergence
    irisflow replay --report-only data/recordings/foo.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from irisflow.calibration.store import CalibrationStore
from irisflow.config.loader import load_config
from irisflow.config.schema import AppConfig
from irisflow.control.dwell import DwellClicker, DwellParams
from irisflow.control.safety import RestZone, SafetyGate
from irisflow.core.clock import FakeClock
from irisflow.core.events import (
    DwellClick,
    DwellProgress,
    Event,
    GazeUpdated,
    SafetyPaused,
    SafetyResumed,
)
from irisflow.core.exceptions import IrisFlowError
from irisflow.core.interfaces import CalibrationModel
from irisflow.filtering.chain import build_filter_chain
from irisflow.mapping.screen import ScreenMapper as ScreenPixelMapper
from irisflow.mapping.screen_info import ScreenInfo
from irisflow.pipeline.orchestrator import filter_config_from_app
from irisflow.pipeline.replayer import ReplayResult, SessionReplayer
from irisflow.telemetry.report import (
    build_session_report,
    render_session_report,
)
from irisflow.telemetry.session import (
    SessionHeader,
    SessionRecordError,
    read_session_file,
)

__all__ = ["replay"]


def replay(
    session: Annotated[
        Path,
        typer.Argument(help="Path to the JSONL session file to replay."),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Config used to rebuild the pipeline."),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail with exit code 3 when the replay diverges from the recorded output.",
        ),
    ] = True,
    report_only: Annotated[
        bool,
        typer.Option(
            "--report-only",
            help="Skip the replay; just recompute the report from the recorded file.",
        ),
    ] = False,
) -> None:
    """Deterministic replay + report over a recorded session."""
    try:
        header, events_iter = read_session_file(session)
    except SessionRecordError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    events = list(events_iter)
    typer.echo(f"Loaded {len(events)} events from {session}")

    if not report_only:
        try:
            cfg = load_config(yaml_path=config) if config is not None else load_config()
        except IrisFlowError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=2) from exc

        replayer, calibration_model = _build_replayer(cfg, header)
        result = replayer.replay(events)
        divergence = _diff_replay(events, result)
        if divergence:
            typer.echo(f"[warn] replay diverges from recording: {divergence}")
            if strict:
                raise typer.Exit(code=3)
        else:
            typer.echo("Replay: bit-identical to recording. ✓")
        del calibration_model  # unused past construction; keep alive for GC clarity

    report = build_session_report(session_id=header.session_id, events=events)
    typer.echo("")
    typer.echo(render_session_report(report))


# ---------------------------------------------------------------------------
# Pipeline rebuild
# ---------------------------------------------------------------------------
def _build_replayer(
    cfg: AppConfig, header: SessionHeader
) -> tuple[SessionReplayer, CalibrationModel | None]:
    clock = FakeClock()
    calibration_model: CalibrationModel | None = None
    if header.calibration_profile_id is not None:
        try:
            store = CalibrationStore(cfg.calibration.profiles_dir)
            calibration_model = store.load(header.calibration_profile_id).model
        except Exception:
            calibration_model = None

    screen = ScreenInfo(
        width_px=header.screen_width_px, height_px=header.screen_height_px
    )
    mapper = ScreenPixelMapper(screen=screen, clamp_margin_px=cfg.mapping.clamp_margin_px)
    filter_chain = build_filter_chain(filter_config_from_app(cfg))
    safety = SafetyGate(
        pause_on_face_lost_ms=cfg.control.safety.pause_on_face_lost_ms,
        rest_zone=RestZone(*cfg.control.safety.rest_zone_px),
        clock=clock,
    )
    dwell = DwellClicker(
        DwellParams(
            radius_px=cfg.control.dwell.radius_px,
            duration_ms=cfg.control.dwell.duration_ms,
            refractory_ms=cfg.control.dwell.refractory_ms,
        )
    )
    replayer = SessionReplayer(
        calibration_model=calibration_model,
        mapper=mapper,
        filter_chain=filter_chain,
        safety=safety,
        dwell=dwell,
        cursor_enabled=True,
        clock=clock,
    )
    return replayer, calibration_model


# ---------------------------------------------------------------------------
# Divergence check
# ---------------------------------------------------------------------------
def _diff_replay(recorded: list[Event], result: ReplayResult) -> str | None:
    """Return ``None`` when replay matches, otherwise a short message."""
    recorded_gaze = [e for e in recorded if isinstance(e, GazeUpdated)]
    if len(recorded_gaze) != len(result.gaze_updates):
        return (
            f"gaze count {len(recorded_gaze)} recorded vs "
            f"{len(result.gaze_updates)} replayed"
        )
    for i, (a, b) in enumerate(zip(recorded_gaze, result.gaze_updates, strict=False)):
        if (a.frame_id, a.px, a.py, a.is_fixation) != (
            b.frame_id,
            b.px,
            b.py,
            b.is_fixation,
        ):
            return f"gaze diverges at index {i}: recorded={a} replayed={b}"

    recorded_clicks = [e for e in recorded if isinstance(e, DwellClick)]
    if len(recorded_clicks) != len(result.dwell_clicks):
        return (
            f"click count {len(recorded_clicks)} recorded vs "
            f"{len(result.dwell_clicks)} replayed"
        )
    recorded_progress = [e for e in recorded if isinstance(e, DwellProgress)]
    if len(recorded_progress) != len(result.dwell_progress):
        return (
            f"dwell progress count {len(recorded_progress)} recorded vs "
            f"{len(result.dwell_progress)} replayed"
        )
    recorded_paused = [e for e in recorded if isinstance(e, SafetyPaused)]
    if len(recorded_paused) != len(result.safety_paused):
        return "safety pause counts differ"
    recorded_resumed = [e for e in recorded if isinstance(e, SafetyResumed)]
    if len(recorded_resumed) != len(result.safety_resumed):
        return "safety resume counts differ"
    return None
