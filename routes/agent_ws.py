"""Agent run timeline websocket APIs."""

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from agent.agent_store import get_agent_store
from auth.auth_service import validate_api_key_for_current_server
from services.chat.ws_manager import get_ws_manager

from .deps import is_websocket_from_tunnel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent-ws"])
_ws_manager = get_ws_manager()

# Keepalive interval (seconds) for idle agent timeline websocket connections.
# Server emits {"type": "ping"} on idle to keep intermediate proxies (tunnels,
# load balancers) from closing the socket. Client need not respond.
_AGENT_WS_IDLE_PING_INTERVAL = 25.0


@router.websocket("/ws/agent/runs/{run_id}")
async def agent_run_timeline_websocket(
    websocket: WebSocket,
    run_id: str,
    api_key: Optional[str] = Query(None),
) -> None:
    """Stream durable events for one agent run."""
    validation = validate_api_key_for_current_server(api_key)
    if not validation.success or (is_websocket_from_tunnel(websocket) and not api_key):
        await websocket.close(code=1008, reason=validation.error or "Invalid API key")
        return

    can_connect, reject_reason = _ws_manager.can_connect(websocket)
    if not can_connect:
        await websocket.close(code=1008, reason=reject_reason)
        return

    store = get_agent_store()
    run = store.get_run(run_id)
    if run is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": f"Agent run '{run_id}' not found"})
        await websocket.close()
        return

    await websocket.accept()
    _ws_manager.register_connection(websocket)
    last_sequence = 0
    last_send_at = time.monotonic()
    try:
        await websocket.send_json(
            {
                "type": "run_snapshot",
                "run": run,
                "messages": store.list_messages(run_id),
                "artifacts": store.list_artifacts(run_id),
            }
        )
        last_send_at = time.monotonic()
        while True:
            events = store.list_events(run_id, after_sequence=last_sequence, limit=100)
            for event in events:
                last_sequence = max(last_sequence, int(event.get("sequence") or 0))
                await websocket.send_json({"type": "agent_event", "event": event})
                last_send_at = time.monotonic()
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                if message == "ping":
                    await websocket.send_json({"type": "pong"})
                    last_send_at = time.monotonic()
            except asyncio.TimeoutError:
                # Idle keepalive: emit a server-side ping to keep proxies/tunnels
                # from closing the socket. Client is not expected to respond.
                if time.monotonic() - last_send_at >= _AGENT_WS_IDLE_PING_INTERVAL:
                    await websocket.send_json({"type": "ping"})
                    last_send_at = time.monotonic()
                continue
    except WebSocketDisconnect:
        logger.info("agent run websocket disconnected run_id=%s", run_id)
    finally:
        _ws_manager.unregister_connection(websocket)
