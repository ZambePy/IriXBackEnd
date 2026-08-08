"""Sprint 12 — HTTP surface: /health, /config, /profiles, /calibration."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from irisflow.api import create_app
from irisflow.calibration.models import AffineCalibration
from irisflow.calibration.store import CalibrationProfile, CalibrationStore
from irisflow.config.schema import AppConfig, CalibrationConfig
from irisflow.core.types import RawGaze
from tests.fixtures.api_helpers import build_stubbed_app_state


@pytest.fixture
def stub_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        calibration=CalibrationConfig(profiles_dir=tmp_path / "profiles"),
    )


@pytest.fixture
def client_with_state(stub_config: AppConfig):
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client:
        yield client, bundle
    bundle.state.stop()


def test_health_reports_starting_or_ok(client_with_state) -> None:
    client, _ = client_with_state
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "starting"}
    assert body["backend"] == "keras"
    assert body["protocol_version"] >= 1
    assert body["pipeline_state"] in {"IDLE", "TRACKING", "LOST"}


def test_config_get_returns_full_tree(client_with_state) -> None:
    client, _ = client_with_state
    resp = client.get("/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "config" in body
    assert body["config"]["control"]["dwell"]["radius_px"] > 0


def test_config_patch_accepts_dwell_tweaks(client_with_state) -> None:
    client, bundle = client_with_state
    resp = client.patch("/config", json={"control.dwell.radius_px": 77})
    assert resp.status_code == 200
    assert bundle.state.config.control.dwell.radius_px == 77


def test_config_patch_rejects_unmutable_field(client_with_state) -> None:
    client, _ = client_with_state
    resp = client.patch("/config", json={"camera.width": 1024})
    assert resp.status_code == 400
    assert "not mutable" in resp.json()["detail"]


def test_config_patch_rejects_empty_body(client_with_state) -> None:
    client, _ = client_with_state
    resp = client.patch("/config", json={})
    assert resp.status_code == 400


def test_profiles_list_reads_from_disk(stub_config: AppConfig) -> None:
    store = CalibrationStore(stub_config.calibration.profiles_dir)
    model = AffineCalibration(profile_id="alice")
    model.fit(
        [
            RawGaze(x=0.0, y=0.0, confidence=1.0, inference_ms=1.0),
            RawGaze(x=1.0, y=0.0, confidence=1.0, inference_ms=1.0),
            RawGaze(x=0.0, y=1.0, confidence=1.0, inference_ms=1.0),
            RawGaze(x=1.0, y=1.0, confidence=1.0, inference_ms=1.0),
        ],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    )
    profile = CalibrationProfile(
        profile_id="alice",
        model=model,
        screen_width=1920,
        screen_height=1080,
        created_at_wall_s=1.0,
        mean_error_px=42.0,
        p95_error_px=55.0,
        max_error_px=70.0,
        n_samples=100,
    )
    store.save(profile)

    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client:
        resp = client.get("/profiles")
    bundle.state.stop()

    assert resp.status_code == 200
    payload = resp.json()
    ids = [p["profile_id"] for p in payload["profiles"]]
    assert "alice" in ids


def test_calibration_status_idle_by_default(client_with_state) -> None:
    client, _ = client_with_state
    resp = client.get("/calibration/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["phase"] == "idle"


def test_calibration_abort_when_none_is_409(client_with_state) -> None:
    client, _ = client_with_state
    resp = client.post("/calibration/abort")
    assert resp.status_code == 409
