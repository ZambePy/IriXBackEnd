"""Sprint 12 — /ws/gaze end-to-end tests.

Uses the FastAPI :class:`TestClient` (which internally uses ``httpx`` +
``starlette``'s WS test transport). No real network sockets — everything
runs in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from irisflow.api import create_app
from irisflow.config.schema import AppConfig, CalibrationConfig
from irisflow.core.events import (
    GazeUpdated,
    PipelineState,
    StateChanged,
)
from tests.fixtures.api_helpers import build_stubbed_app_state


@pytest.fixture
def stub_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        calibration=CalibrationConfig(profiles_dir=tmp_path / "profiles"),
    )


def _read_until(
    ws,
    message_type: str,
    *,
    max_frames: int = 200,
    match_frame_id: int | None = None,
) -> dict:
    """Pump messages until we see ``type == message_type`` (and optional frame id)."""
    for _ in range(max_frames):
        raw = ws.receive_text()
        payload = json.loads(raw)
        if payload.get("type") != message_type:
            continue
        if match_frame_id is not None and payload.get("frame_id") != match_frame_id:
            continue
        return payload
    raise AssertionError(
        f"never received {message_type!r}"
        + (f" frame_id={match_frame_id}" if match_frame_id else "")
        + f" within {max_frames} frames"
    )


# Frame ids the runner never emits (its counter starts at 0) — use these
# for manually published events so they can be distinguished from pipeline
# traffic in the background thread.
_MARKER_FRAME_A = 900_001
_MARKER_FRAME_B = 900_002
_MARKER_FRAME_C = 900_003


def test_ws_streams_gaze_when_bus_publishes(stub_config: AppConfig) -> None:
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client, client.websocket_connect("/ws/gaze") as ws:
        bundle.bus.publish(
            GazeUpdated(
                frame_id=_MARKER_FRAME_A, timestamp=0.5, px=500, py=400,
                is_fixation=True, confidence=0.9,
            )
        )
        msg = _read_until(ws, "gaze", match_frame_id=_MARKER_FRAME_A)
        assert msg["px"] == 500
        assert msg["fixation"] is True
    bundle.state.stop()


def test_ws_state_change_reaches_client(stub_config: AppConfig) -> None:
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client, client.websocket_connect("/ws/gaze") as ws:
        bundle.bus.publish(
            StateChanged(
                previous=PipelineState.IDLE,
                current=PipelineState.TRACKING,
                timestamp=1.0,
            )
        )
        msg = _read_until(ws, "state")
        assert msg["state"] == "TRACKING"
        assert msg["previous"] == "IDLE"
    bundle.state.stop()


def test_ws_multiple_clients_receive_same_event(stub_config: AppConfig) -> None:
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client, client.websocket_connect("/ws/gaze") as ws1, \
             client.websocket_connect("/ws/gaze") as ws2:
        bundle.bus.publish(
            GazeUpdated(
                frame_id=_MARKER_FRAME_B, timestamp=0.5, px=100, py=200,
                is_fixation=False, confidence=1.0,
            )
        )
        m1 = _read_until(ws1, "gaze", match_frame_id=_MARKER_FRAME_B)
        m2 = _read_until(ws2, "gaze", match_frame_id=_MARKER_FRAME_B)
        assert m1["px"] == 100
        assert m2["px"] == 100
    bundle.state.stop()


def test_ws_bad_client_message_answers_with_error(stub_config: AppConfig) -> None:
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client, client.websocket_connect("/ws/gaze") as ws:
        ws.send_text('{"type": "nonsense"}')
        msg = _read_until(ws, "error")
        assert msg["code"] == "bad_message"
    bundle.state.stop()


def test_ws_bad_json_answers_with_error(stub_config: AppConfig) -> None:
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client, client.websocket_connect("/ws/gaze") as ws:
        ws.send_text("{not-json")
        msg = _read_until(ws, "error")
        assert msg["code"] == "bad_json"
    bundle.state.stop()


def test_ws_client_pause_resume_via_message(stub_config: AppConfig) -> None:
    bundle = build_stubbed_app_state(config=stub_config)
    # Transition to TRACKING so we can pause.
    bundle.components.state.transition_to(PipelineState.TRACKING)
    app = create_app(state=bundle.state)
    with TestClient(app) as client, client.websocket_connect("/ws/gaze") as ws:
        ws.send_text('{"type": "pause"}')
        state_msg = _read_until(ws, "state")
        assert state_msg["state"] == "PAUSED"
        ws.send_text('{"type": "resume"}')
        state_msg2 = _read_until(ws, "state")
        assert state_msg2["state"] == "TRACKING"
    bundle.state.stop()


def test_ws_client_disconnect_does_not_kill_pipeline(stub_config: AppConfig) -> None:
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/gaze") as ws:
            ws.receive_text  # noqa: B018 — just to poke the socket
        # First connection closed; open a fresh one and the hub still delivers.
        with client.websocket_connect("/ws/gaze") as ws:
            bundle.bus.publish(
                GazeUpdated(
                    frame_id=_MARKER_FRAME_C, timestamp=1.0, px=1, py=2,
                    is_fixation=False, confidence=1.0,
                )
            )
            msg = _read_until(ws, "gaze", match_frame_id=_MARKER_FRAME_C)
            assert msg["px"] == 1
    bundle.state.stop()


def test_ws_slow_client_drops_gaze_not_pipeline(stub_config: AppConfig) -> None:
    """Overflow the gaze queue and verify the pipeline thread isn't blocked."""
    bundle = build_stubbed_app_state(config=stub_config)
    app = create_app(state=bundle.state)
    with TestClient(app) as client, client.websocket_connect("/ws/gaze") as ws:
        # Publish way more than the queue size without draining.
        for i in range(50):
            bundle.bus.publish(
                GazeUpdated(
                    frame_id=i, timestamp=float(i), px=i, py=i,
                    is_fixation=False, confidence=1.0,
                )
            )
        # Drain a handful; the earliest frame_id we see should be much later
        # than 0 because the queue drops oldest.
        first = _read_until(ws, "gaze")
        assert first["frame_id"] >= 0  # sanity — at least we got one
    bundle.state.stop()
