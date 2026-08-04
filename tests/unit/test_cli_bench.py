"""Sprint 6 — ``irisflow bench`` command surface."""

from __future__ import annotations

from typer.testing import CliRunner

from irisflow.cli.main import app

runner = CliRunner()


def test_bench_appears_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "bench" in result.stdout


def test_bench_requires_at_least_one_mode() -> None:
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "at least one" in combined.lower()


def test_bench_help_lists_modes() -> None:
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "--latency" in result.stdout
    assert "--compare-backends" in result.stdout
    assert "--sanity-check" in result.stdout
