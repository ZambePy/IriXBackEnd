"""``/ws/gaze`` — bidirectional WebSocket endpoint (Sprint 12).

Wire protocol summary:

* On connect the server sends nothing until data starts flowing on the
  bus. Clients may push messages immediately.
* Server → client: a discriminated union :class:`ServerMessage` — see
  :mod:`irisflow.api.schemas`.
* Client → server: :class:`ClientMessage` — every message is validated
  against the Pydantic union; unknown ``type`` values are answered with
  an :class:`ErrorMessage`.
* Disconnect is graceful on either side; the server tolerates the client
  vanishing without a close frame.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from irisflow.api.schemas import (
    CalibrationAbortMessage,
    CalibrationReadyMessage,
    CalibrationResultMessage,
    ClientMessage,
    ErrorMessage,
    PauseMessage,
    ResumeMessage,
    ServerMessage,
    SetCursorControlMessage,
    SetProfileMessage,
    StartCalibrationMessage,
    StartTrackingMessage,
)
from irisflow.api.session import ClientSession
from irisflow.api.state import AppState
from irisflow.core.events import CalibrationProgress, Event
from irisflow.core.exceptions import CalibrationError, IrisFlowError

router = APIRouter()


_client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


@router.websocket("/ws/gaze")
async def gaze_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    state: AppState | None = getattr(websocket.app.state, "irisflow", None)
    if state is None:
        await websocket.close(code=1011, reason="pipeline not initialized")
        return
    loop = asyncio.get_running_loop()
    session = ClientSession(loop=loop)
    state.hub.register(session)

    async def _sender() -> None:
        try:
            while True:
                message = await session.next_outgoing()
                await websocket.send_text(message.model_dump_json())
        except (WebSocketDisconnect, RuntimeError):
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _receiver() -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                await _handle_client_text(state, websocket, raw)
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    send_task = asyncio.create_task(_sender())
    recv_task = asyncio.create_task(_receiver())
    try:
        _done, pending = await asyncio.wait(
            {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    finally:
        state.hub.unregister(session)
        with contextlib.suppress(Exception):
            await websocket.close()


async def _handle_client_text(
    state: AppState, websocket: WebSocket, raw: str
) -> None:
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError:
        await _send(websocket, ErrorMessage(code="bad_json", message="payload must be JSON"))
        return
    try:
        message = _client_adapter.validate_python(payload)
    except ValidationError as exc:
        await _send(
            websocket,
            ErrorMessage(code="bad_message", message=exc.errors(include_url=False)[0]["msg"]),
        )
        return
    try:
        await _dispatch(state, websocket, message)
    except IrisFlowError as exc:
        await _send(websocket, ErrorMessage(code="pipeline_error", message=str(exc)))


async def _dispatch(
    state: AppState, websocket: WebSocket, message: ClientMessage
) -> None:
    if isinstance(message, StartTrackingMessage):
        state.request_pipeline_resume()
        return
    if isinstance(message, PauseMessage):
        state.request_pipeline_pause()
        return
    if isinstance(message, ResumeMessage):
        state.request_pipeline_resume()
        return
    if isinstance(message, SetCursorControlMessage):
        if message.enabled:
            state.components.cursor.enable() if state.components.cursor else None
        else:
            state.components.cursor.disable() if state.components.cursor else None
        return
    if isinstance(message, SetProfileMessage):
        try:
            state.set_active_profile(message.profile)
        except IrisFlowError as exc:
            await _send(websocket, ErrorMessage(code="profile_not_found", message=str(exc)))
        return
    if isinstance(message, StartCalibrationMessage):
        await _handle_start_calibration(state, websocket, message)
        return
    if isinstance(message, CalibrationReadyMessage):
        coord = state.calibration
        if coord is None:
            await _send(
                websocket,
                ErrorMessage(code="no_calibration", message="no calibration in progress"),
            )
            return
        try:
            coord.request_target(message.index)
        except CalibrationError as exc:
            await _send(websocket, ErrorMessage(code="bad_target", message=str(exc)))
        return
    if isinstance(message, CalibrationAbortMessage):
        state.stop_calibration()
        return


async def _handle_start_calibration(
    state: AppState, websocket: WebSocket, message: StartCalibrationMessage
) -> None:
    def _publish(event: Event) -> None:
        # The CalibrationSession publisher signature is ``CalibrationProgress``,
        # not ``Event`` — reuse the bus so the SessionHub picks it up and
        # every connected client sees it.
        if isinstance(event, CalibrationProgress):
            state.components.bus.publish(event)

    def _broadcast_result(result_msg: CalibrationResultMessage) -> None:
        # Send to every connected client via the hub.
        for session in list(state.hub._sessions):
            session.offer(result_msg)

    try:
        state.start_calibration(
            profile_id=message.profile_id,
            points=message.points,
            screen_width=message.screen.w,
            screen_height=message.screen.h,
            publisher=_publish,
            broadcast_result=_broadcast_result,
        )
    except CalibrationError as exc:
        await _send(websocket, ErrorMessage(code="calibration_error", message=str(exc)))


async def _send(websocket: WebSocket, message: ServerMessage) -> None:
    try:
        await websocket.send_text(message.model_dump_json())
    except (WebSocketDisconnect, RuntimeError):
        return


__all__ = ["router"]
