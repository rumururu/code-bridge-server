"""Browser handoff streaming websocket routes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from agent.agent_store import get_agent_store
from agent.browser_session_store import get_browser_session_store
from agent.browser_stream_session import (
    BrowserStreamUnavailable,
    PlaywrightBrowserStreamSession,
)
from auth.auth_service import validate_api_key_for_current_server
from services.chat.ws_manager import get_ws_manager

from .deps import is_websocket_from_tunnel, start_periodic_reauth_task

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent-browser-stream"])
_ws_manager = get_ws_manager()


@router.websocket("/ws/agent/tasks/{task_id}/browser-handoff/stream")
async def browser_handoff_stream_websocket(
    websocket: WebSocket,
    task_id: str,
    api_key: Optional[str] = Query(None),
    width: int = Query(1280, ge=320, le=3840),
    height: int = Query(720, ge=240, le=2160),
    fps: float = Query(2.0, ge=0.5, le=8.0),
    image_type: str = Query("jpeg"),
    quality: int = Query(70, ge=20, le=95),
) -> None:
    """Stream the active browser handoff page and accept user input events."""
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
                "type": "browser_stream.error",
                "code": "no_active_browser_handoff",
                "message": "Task has no active browser handoff.",
            }
        )
        await websocket.close()
        return

    session = payload["browser_session"]
    run_id = str(session["run_id"])
    stream = PlaywrightBrowserStreamSession(
        session,
        viewport_width=width,
        viewport_height=height,
    )
    _ws_manager.register_connection(websocket)
    reauth_task = start_periodic_reauth_task(websocket, api_key)
    frame_interval = 1.0 / max(0.5, min(float(fps), 8.0))
    last_frame_at = 0.0
    store = get_agent_store()

    try:
        await stream.start()
        store.append_event(
            run_id=run_id,
            event_type="task.browser_handoff.stream.connected",
            app_event={"task_id": task_id, "browser_session_id": session["id"]},
        )
        await websocket.send_json(
            {
                "type": "browser_stream.ready",
                "task_id": task_id,
                "browser_session": stream.browser_session,
                "width": width,
                "height": height,
                "fps": fps,
            }
        )
        while True:
            now = time.monotonic()
            if now - last_frame_at >= frame_interval:
                await websocket.send_json(
                    {
                        "type": "browser_stream.frame",
                        "task_id": task_id,
                        "browser_session_id": session["id"],
                        "timestamp": time.time(),
                        **(
                            await stream.snapshot(
                                image_type=image_type,
                                quality=quality,
                            )
                        ),
                    }
                )
                last_frame_at = now

            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=min(0.25, frame_interval),
                )
            except asyncio.TimeoutError:
                continue

            message = _decode_message(raw)
            if message is None:
                await websocket.send_json(
                    {
                        "type": "browser_stream.input_result",
                        "ok": False,
                        "error": "Invalid JSON message.",
                    }
                )
                continue

            message_type = str(message.get("type") or "").strip()
            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif message_type in {"snapshot", "frame"}:
                await websocket.send_json(
                    {
                        "type": "browser_stream.frame",
                        "task_id": task_id,
                        "browser_session_id": session["id"],
                        "timestamp": time.time(),
                        **(
                            await stream.snapshot(
                                image_type=image_type,
                                quality=quality,
                            )
                        ),
                    }
                )
                last_frame_at = time.monotonic()
            elif message_type == "close":
                await websocket.send_json({"type": "browser_stream.closed"})
                break
            else:
                result = await stream.handle_client_message(message)
                if _is_sensitive_input_message(message):
                    result = {
                        **result,
                        "sensitive": True,
                    }
                await websocket.send_json(result)
    except BrowserStreamUnavailable as exc:
        message = str(exc) or "Browser stream is unavailable."
        await websocket.send_json(
            {
                "type": "browser_stream.error",
                "code": "live_browser_session_missing"
                if message == "live_browser_session_missing"
                else "browser_stream_unavailable",
                "message": message,
            }
        )
    except WebSocketDisconnect:
        logger.info("browser handoff stream disconnected task_id=%s", task_id)
    finally:
        await stream.close()
        store.append_event(
            run_id=run_id,
            event_type="task.browser_handoff.stream.disconnected",
            app_event={"task_id": task_id, "browser_session_id": session["id"]},
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


def _is_sensitive_input_message(message: dict[str, Any]) -> bool:
    message_type = str(message.get("type") or "").strip()
    if message.get("sensitive") is True:
        return True
    if message_type not in {"type_text", "text"}:
        return False
    return any(
        str(message.get(key) or "").lower() in {"password", "otp", "2fa", "credential"}
        for key in ("purpose", "input_kind", "field_type")
    )


def _active_browser_handoff_payload(task_id: str) -> dict[str, Any] | None:
    store = get_agent_store()
    payload = store.get_task_checkpoint(task_id)
    if payload is None or payload.get("checkpoint") is None or payload.get("step") is None:
        return None
    checkpoint = payload.get("checkpoint")
    step = payload.get("step")
    if not isinstance(checkpoint, dict) or not isinstance(step, dict):
        return None
    session_id = checkpoint.get("browser_session_id")
    if not isinstance(session_id, str) or not session_id:
        output = step.get("output")
        if isinstance(output, dict):
            fallback = output.get("browser_session_id")
            if isinstance(fallback, str):
                session_id = fallback
    if not isinstance(session_id, str) or not session_id:
        return None
    session = get_browser_session_store().get(session_id)
    if session is None:
        return None
    if session.get("status") not in {"created", "waiting_for_user", "resumed"}:
        return None
    return {**payload, "browser_session": session}
