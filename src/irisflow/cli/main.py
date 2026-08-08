"""IrisFlow CLI entry point.

Sprint 0 wired only ``--version``; Sprint 3 added ``doctor``; Sprint 4 adds
``preview``. Later sprints add ``calibrate`` (S8), ``run`` (S10),
``bench`` / ``replay`` (S6/S11), ``serve`` (S12).
"""

from __future__ import annotations

import typer

from irisflow import __version__
from irisflow.cli.commands.bench import bench as _bench_command
from irisflow.cli.commands.calibrate import calibrate as _calibrate_command
from irisflow.cli.commands.doctor import doctor as _doctor_command
from irisflow.cli.commands.preview import preview as _preview_command
from irisflow.cli.commands.replay import replay as _replay_command
from irisflow.cli.commands.run import run as _run_command
from irisflow.cli.commands.serve import serve as _serve_command

app = typer.Typer(
    name="irisflow",
    help="IrisFlow — webcam-based gaze tracking backend.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("doctor", help="Enumerate cameras and measure real capture rate.")(_doctor_command)
app.command("preview", help="Overlay face + eye ROIs on the live webcam feed.")(_preview_command)
app.command("bench", help="Inference latency / backend parity / sanity check.")(_bench_command)
app.command("run", help="Fim-a-fim: webcam -> gaze coordinate in the terminal.")(_run_command)
app.command(
    "calibrate",
    help="Run a per-user calibration session and save the fitted profile.",
)(_calibrate_command)
app.command(
    "replay",
    help="Deterministically replay a recorded session and print the report.",
)(_replay_command)
app.command(
    "serve",
    help="Serve the FastAPI + WebSocket API (Sprint 12).",
)(_serve_command)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"irisflow {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the IrisFlow version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Root callback — flags only, no default action."""
