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
    out = result.flat_stdout
    assert "--latency" in out
    assert "--compare-backends" in out
    assert "--sanity-check" in out
    assert "--filters" in out


def test_bench_filters_runs_end_to_end() -> None:
    """`bench --filters` uses pure-python filter chains; it should not
    require the CNN, camera, or ONNX runtime."""
    result = runner.invoke(app, ["bench", "--filters"])
    assert result.exit_code == 0
    assert "jitter_rms_px" in result.stdout
    assert "ramp_lag_ms" in result.stdout
    assert "yaml" in result.stdout
    assert "passthrough" in result.stdout
