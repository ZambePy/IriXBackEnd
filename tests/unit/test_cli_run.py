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
    assert "--no-cursor" in result.stdout
    assert "--cursor" in result.stdout
    assert "--config" in result.stdout
    assert "--metrics-every" in result.stdout
    assert "--quiet-gaze" in result.stdout
    assert "--profile" in result.stdout


def test_run_help_mentions_screen_dimensions() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--screen-width" in result.stdout
    assert "--screen-height" in result.stdout
