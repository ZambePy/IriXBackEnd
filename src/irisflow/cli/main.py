"""IrisFlow CLI entry point.

Sprint 0 wired only ``--version``; Sprint 3 adds ``doctor``. Later sprints
add ``preview`` (S4), ``calibrate`` (S8), ``run`` (S10), ``bench`` /
``replay`` (S6/S11), ``serve`` (S12).
"""

from __future__ import annotations

import typer

from irisflow import __version__
from irisflow.cli.commands.doctor import doctor as _doctor_command

app = typer.Typer(
    name="irisflow",
    help="IrisFlow — webcam-based gaze tracking backend.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("doctor", help="Enumerate cameras and measure real capture rate.")(
    _doctor_command
)


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
