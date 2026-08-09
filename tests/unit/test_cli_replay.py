"""Sprint 11 — ``irisflow replay`` CLI surface + report over a fixture session."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from irisflow.cli.main import app

runner = CliRunner()


FIXTURE = Path(__file__).parent.parent / "fixtures" / "sessions" / "synthetic_session.jsonl"


def test_replay_appears_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "replay" in result.stdout


def test_replay_help_lists_flags() -> None:
    result = runner.invoke(app, ["replay", "--help"])
    assert result.exit_code == 0
    out = result.flat_stdout
    assert "--strict" in out
    assert "--report-only" in out
    assert "--config" in out


def test_report_only_over_fixture_prints_report() -> None:
    result = runner.invoke(app, ["replay", "--report-only", str(FIXTURE)])
    assert result.exit_code == 0, result.stdout
    assert "session       fixture-synth-01" in result.stdout
    assert "fps" in result.stdout
    assert "jitter_rms_px" in result.stdout


def test_replay_missing_file_exits_with_error() -> None:
    result = runner.invoke(app, ["replay", "does-not-exist.jsonl"])
    assert result.exit_code == 2
