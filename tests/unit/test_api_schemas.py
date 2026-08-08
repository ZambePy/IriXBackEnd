"""Sprint 12 — WS wire contracts (:mod:`irisflow.api.schemas`)."""

from __future__ import annotations

import json

from pydantic import TypeAdapter

from irisflow.api.schemas import (
    PROTOCOL_VERSION,
    CalibrationReadyMessage,
    ClientMessage,
    GazeMessage,
    ServerMessage,
    StartCalibrationMessage,
    StateMessage,
)

_client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
_server_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def test_gaze_message_serializes_with_discriminator() -> None:
    msg = GazeMessage(
        frame_id=1,
        ts=1.23,
        px=100,
        py=200,
        nx=0.1,
        ny=0.2,
        fixation=True,
        confidence=0.9,
    )
    payload = json.loads(msg.model_dump_json())
    assert payload["type"] == "gaze"
    assert payload["px"] == 100

    round_trip = _server_adapter.validate_python(payload)
    assert isinstance(round_trip, GazeMessage)


def test_state_and_client_messages_discriminate_by_type() -> None:
    state = _server_adapter.validate_python(
        {"type": "state", "ts": 0.0, "state": "TRACKING", "previous": "IDLE"}
    )
    assert isinstance(state, StateMessage)

    ready = _client_adapter.validate_python({"type": "calibration_ready", "index": 3})
    assert isinstance(ready, CalibrationReadyMessage)

    start = _client_adapter.validate_python(
        {
            "type": "start_calibration",
            "points": 9,
            "screen": {"w": 1920, "h": 1080},
            "profile_id": "maria",
        }
    )
    assert isinstance(start, StartCalibrationMessage)


def test_protocol_version_exposed() -> None:
    assert PROTOCOL_VERSION >= 1
