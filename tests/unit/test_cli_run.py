"""Sprint 10 — ``irisflow run`` command surface (no hardware / no model)."""

from __future__ import annotations

from typer.testing import CliRunner

from irisflow.cli.main import app

runner = CliRunner()


def test_run_appears_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_run_help_exposes_flags() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = result.flat_stdout
    assert "--no-cursor" in out
    assert "--cursor" in out
    assert "--config" in out
    assert "--metrics-every" in out
    assert "--quiet-gaze" in out
    assert "--profile" in out


def test_run_help_mentions_screen_dimensions() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = result.flat_stdout
    assert "--screen-width" in out
    assert "--screen-height" in out
