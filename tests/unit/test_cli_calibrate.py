"""Sprint 8 -- ``irisflow calibrate`` command surface.

Only smoke coverage: the command opens a real webcam and loads TF,
neither of which fits an offline test. We assert the command exists,
takes the expected options, and refuses to run when config is invalid.
"""

from __future__ import annotations

from typer.testing import CliRunner

from irisflow.cli.main import app

runner = CliRunner()


def test_calibrate_help_lists_profile_option() -> None:
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    out = result.flat_stdout
    assert "--profile" in out
    assert "--screen-width" in out


def test_calibrate_requires_profile() -> None:
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code != 0
