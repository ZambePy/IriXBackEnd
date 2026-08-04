"""Sprint 3 — ``irisflow doctor`` command.

The doctor command must:

* Report clearly when no camera is available (exit code != 0, message
  telling the user what to check).
* Print an enumeration table and a measured FPS block when a device does
  open — this is the smoke check the developer runs after wiring a camera.

Both cases are exercised here by monkey-patching :class:`cv2.VideoCapture`
so the test suite never depends on real hardware (SPRINTS.md §3 criteria).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from typer.testing import CliRunner

from irisflow.cli.main import app
from tests.fixtures.fake_capture import FakeVideoCapture, make_factory

runner = CliRunner()


@pytest.fixture
def doctor_config(tmp_path: Path) -> Path:
    """Minimal YAML pointing at device_id 0 with tiny frame size for speed."""
    cfg = tmp_path / "doctor.yaml"
    cfg.write_text(
        "camera:\n"
        "  device_id: 0\n"
        "  width: 320\n"
        "  height: 240\n"
        "  fps: 30\n"
        "  reconnect_backoff_ms: 10\n",
        encoding="utf-8",
    )
    return cfg


def test_doctor_reports_no_camera_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch, doctor_config: Path
) -> None:
    def _never_opens(device_id: int) -> FakeVideoCapture:
        return FakeVideoCapture(device_id, open_ok=False)

    monkeypatch.setattr(cv2, "VideoCapture", _never_opens)

    result = runner.invoke(
        app,
        ["doctor", "--config", str(doctor_config), "--max-index", "2", "--duration", "0.1"],
    )
    assert result.exit_code != 0
    assert "No camera devices detected" in result.stdout


def test_doctor_measures_configured_camera_when_available(
    monkeypatch: pytest.MonkeyPatch, doctor_config: Path
) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", make_factory(width=320, height=240, fps=30.0))

    result = runner.invoke(
        app,
        ["doctor", "--config", str(doctor_config), "--max-index", "2", "--duration", "0.3"],
    )
    assert result.exit_code == 0, result.stdout
    assert "Measured capture" in result.stdout
    assert "measured FPS" in result.stdout
    assert "frames captured" in result.stdout


def test_doctor_help_lists_the_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout
