"""Browser handoff WebRTC websocket routes."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from agent.agent_store import get_agent_store
from agent.browser_rtc_session import BrowserRtcPeerSession, BrowserRtcUnavailable
from auth.auth_service import validate_api_key_for_current_server
from services.chat.ws_manager import get_ws_manager

from .agent_browser_stream import (
    _active_browser_handoff_payload,
    _is_sensitive_input_message,
)
from .deps import is_websocket_from_tunnel, start_periodic_reauth_task

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent-browser-rtc"])
_ws_manager = get_ws_manager()


@router.websocket("/ws/agent/tasks/{task_id}/browser-handoff/rtc")
async def browser_handoff_rtc_websocket(
    websocket: WebSocket,
    task_id: str,
    api_key: Optional[str] = Query(None),
    width: int = Query(1280, ge=320, le=3840),
    height: int = Query(720, ge=240, le=2160),
    fps: float = Query(12.0, ge=1.0, le=30.0),
    quality: int = Query(70, ge=20, le=95),
    capture: str = Query("page", pattern="^page$"),
) -> None:
    """Signal a WebRTC browser handoff stream and accept DataChannel input."""
    validation = validate_api_key_for_current_server(api_key)
    if not validation.success or (is_websocket_from_tunnel(websocket) and not api_key):
        await websocket.close(code=1008, reason=validation.error or "Invalid API key")
        return

    can_connect, reject_reason = _ws_manager.can_connect(websocket)
    if not can_connect:
        await websocket.close(code=1008, reason=reject_reason)
        return

    payload = _active_browser_handoff_payload(task_id)
    await websocket.accept()
    if payload is None:
        await websocket.send_json(
            {
                "type": "browser_rtc.error",
                "code": "no_active_browser_handoff",
                "message": "Task has no active browser handoff.",
            }
        )
        await websocket.close()
        return

    session = payload["browser_session"]
    run_id = str(session["run_id"])
    store = get_agent_store()

    async def redact_result(
        message: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if not _is_sensitive_input_message(message):
            return result
        return {**result, "sensitive": True}

    rtc = BrowserRtcPeerSession(
        session,
        viewport_width=width,
        viewport_height=height,
        fps=fps,
        quality=quality,
        capture_mode=capture,
        input_result_callback=redact_result,
    )
    _ws_manager.register_connection(websocket)
    reauth_task = start_periodic_reauth_task(websocket, api_key)

    try:
        await rtc.start()
        store.append_event(
            run_id=run_id,
            event_type="task.browser_handoff.rtc.connected",
            app_event={
                "task_id": task_id,
                "browser_session_id": session["id"],
                "capture": capture,
            },
        )
        await websocket.send_json(
            {
                "type": "browser_rtc.ready",
                "task_id": task_id,
                "browser_session": rtc.browser_session,
                "width": width,
                "height": height,
                "fps": fps,
                "capture": capture,
            }
        )
        while True:
            raw = await websocket.receive_text()
            message = _decode_message(raw)
            if message is None:
                await websocket.send_json(
                    {
                        "type": "browser_rtc.error",
                        "code": "invalid_json",
                        "message": "Invalid JSON message.",
                    }
                )
                continue

            message_type = str(message.get("type") or "").strip()
            if message_type == "offer":
                answer = await rtc.accept_offer(
                    sdp=str(message.get("sdp") or ""),
                    offer_type=str(message.get("sdpType") or "offer"),
                )
                await websocket.send_json(
                    {
                        "type": "answer",
                        "sdp": answer["sdp"],
                        "sdpType": answer["type"],
                    }
                )
            elif message_type == "candidate":
                candidate = message.get("candidate")
                if isinstance(candidate, dict):
                    await rtc.add_ice_candidate(candidate)
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif message_type == "close":
                await websocket.send_json({"type": "browser_rtc.closed"})
                break
            else:
                await websocket.send_json(
                    {
                        "type": "browser_rtc.error",
                        "code": "unsupported_message",
                        "message": f"Unsupported RTC signal: {message_type}",
                    }
                )
    except BrowserRtcUnavailable as exc:
        message = str(exc) or "Browser RTC stream is unavailable."
        await websocket.send_json(
            {
                "type": "browser_rtc.error",
                "code": "live_browser_session_missing"
                if message == "live_browser_session_missing"
                else "browser_rtc_unavailable",
                "message": message,
            }
        )
    except WebSocketDisconnect:
        logger.info("browser handoff rtc disconnected task_id=%s", task_id)
    finally:
        await rtc.close()
        store.append_event(
            run_id=run_id,
            event_type="task.browser_handoff.rtc.disconnected",
            app_event={
                "task_id": task_id,
                "browser_session_id": session["id"],
                "capture": capture,
            },
        )
        if reauth_task is not None and not reauth_task.done():
            reauth_task.cancel()
        _ws_manager.unregister_connection(websocket)


def _decode_message(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
