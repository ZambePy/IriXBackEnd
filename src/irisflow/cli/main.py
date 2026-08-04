"""IrisFlow CLI entry point.

Sprint 0 wires only ``--version`` — later sprints add ``doctor`` (S3),
``preview`` (S4), ``calibrate`` (S8), ``run`` (S10), ``bench`` / ``replay``
(S6/S11), ``serve`` (S12).
"""

from __future__ import annotations

import typer

from irisflow import __version__

app = typer.Typer(
    name="irisflow",
    help="IrisFlow — webcam-based gaze tracking backend.",
    no_args_is_help=True,
    add_completion=False,
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
