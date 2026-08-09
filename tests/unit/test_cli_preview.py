"""Sprint 4 — ``irisflow preview`` command surface.

Preview requires a real display and MediaPipe model, so we only assert
its CLI wiring (help + failure mode when the model asset is missing).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from irisflow.cli.main import app

runner = CliRunner()


def test_preview_is_listed_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "preview" in result.stdout


def test_preview_help_renders_options() -> None:
    result = runner.invoke(app, ["preview", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.flat_stdout


def test_preview_exits_gracefully_when_model_missing(tmp_path: Path) -> None:
    cfg = tmp_path / "preview.yaml"
    missing_model = tmp_path / "missing.task"
    cfg.write_text(
        f"detection:\n  face_model_path: {missing_model.as_posix()}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["preview", "--config", str(cfg)])
    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "face landmarker model not found" in combined.lower() or (
        "not found" in combined.lower()
    )


@pytest.mark.parametrize("bad_flag", ["--nope", "--unknown-thing"])
def test_preview_rejects_unknown_flags(bad_flag: str) -> None:
    result = runner.invoke(app, ["preview", bad_flag])
    assert result.exit_code != 0
