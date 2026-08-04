"""Sprint 0: minimal CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from irisflow import __version__
from irisflow.cli.main import app

runner = CliRunner()


def test_version_flag_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_short_version_flag_matches_long_form() -> None:
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help_and_exits_nonzero() -> None:
    """`no_args_is_help=True` — bare `irisflow` prints help and returns non-zero."""
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "IrisFlow" in result.stdout
