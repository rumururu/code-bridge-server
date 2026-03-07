"""WebSocket chat routes for LLM streaming."""

import json
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from chat_stream_service import stream_claude_turn
from chat_ws_service import (
    create_chat_session_for_current_server,
    process_disconnect_server_message_for_current_server,
    process_firebase_auth_message_for_current_server,
    resolve_chat_provider_selection_for_current_server,
    validate_chat_websocket_access_for_current_server,
)

router = APIRouter(tags=["chat"])

# WebSocket connection limits
MAX_CONNECTIONS_PER_IP = 5  # Max concurrent connections per IP
MAX_MESSAGES_PER_MINUTE = 30  # Max messages per minute per connection
CONNECTION_RATE_LIMIT_WINDOW = 60  # Seconds
CONNECTION_RATE_LIMIT_MAX = 10  # Max connection attempts per window


class WebSocketConnectionManager:
    """Manages WebSocket connection limits and rate limiting."""

    def __init__(self):
        self._connections_per_ip: dict[str, int] = defaultdict(int)
        self._connection_attempts: dict[str, list[float]] = defaultdict(list)
        self._message_timestamps: dict[int, list[float]] = defaultdict(list)

    def _get_client_ip(self, websocket: WebSocket) -> str:
        """Extract client IP from WebSocket connection."""
        # Check Cloudflare header first
        cf_ip = websocket.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip.strip()
        # Fall back to direct client
        if websocket.client:
            return websocket.client.host
        return "unknown"

    def _cleanup_old_attempts(self, client_ip: str) -> None:
        """Remove connection attempts older than the rate limit window."""
        now = time.time()
        cutoff = now - CONNECTION_RATE_LIMIT_WINDOW
        self._connection_attempts[client_ip] = [
            ts for ts in self._connection_attempts[client_ip] if ts > cutoff
        ]

    def can_connect(self, websocket: WebSocket) -> tuple[bool, str]:
        """Check if a new connection is allowed.

        Returns:
            Tuple of (is_allowed, rejection_reason)
        """
        client_ip = self._get_client_ip(websocket)
        now = time.time()

        # Check connection rate limit
        self._cleanup_old_attempts(client_ip)
        if len(self._connection_attempts[client_ip]) >= CONNECTION_RATE_LIMIT_MAX:
            return False, "Too many connection attempts. Please wait."

        # Record connection attempt
        self._connection_attempts[client_ip].append(now)

        # Check concurrent connection limit
        if self._connections_per_ip[client_ip] >= MAX_CONNECTIONS_PER_IP:
            return False, f"Maximum concurrent connections ({MAX_CONNECTIONS_PER_IP}) reached."

        return True, ""

    def register_connection(self, websocket: WebSocket) -> None:
        """Register a new active connection."""
        client_ip = self._get_client_ip(websocket)
        self._connections_per_ip[client_ip] += 1

    def unregister_connection(self, websocket: WebSocket) -> None:
        """Unregister a connection when it closes."""
        client_ip = self._get_client_ip(websocket)
        if self._connections_per_ip[client_ip] > 0:
            self._connections_per_ip[client_ip] -= 1
        # Clean up message timestamps
        ws_id = id(websocket)
        if ws_id in self._message_timestamps:
            del self._message_timestamps[ws_id]

    def can_send_message(self, websocket: WebSocket) -> tuple[bool, str]:
        """Check if a message can be sent (rate limiting).

        Returns:
            Tuple of (is_allowed, rejection_reason)
        """
        ws_id = id(websocket)
        now = time.time()
        cutoff = now - 60  # 1 minute window

        # Clean up old timestamps
        self._message_timestamps[ws_id] = [
            ts for ts in self._message_timestamps[ws_id] if ts > cutoff
        ]

        if len(self._message_timestamps[ws_id]) >= MAX_MESSAGES_PER_MINUTE:
            return False, f"Message rate limit exceeded ({MAX_MESSAGES_PER_MINUTE}/min)."

        self._message_timestamps[ws_id].append(now)
        return True, ""


# Global connection manager
_ws_manager = WebSocketConnectionManager()


async def _handle_user_message(
    websocket: WebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
) -> None:
    user_message_raw = message.get("content", "")
    user_message = str(user_message_raw).strip()
    print(f"[chat_ws] project={project_name} incoming user_message len={len(user_message)}")
    if not user_message:
        await websocket.send_json({"type": "error", "message": "Message content is empty"})
        return

    await websocket.send_json({"type": "user_message", "content": user_message})
    await stream_claude_turn(
        websocket,
        session,
        project_name=project_name,
        user_message=user_message,
    )


async def _handle_approve_permissions(
    websocket: WebSocket,
    session,
    project_name: str,
) -> None:
    if not session.has_pending_permission_denials:
        await websocket.send_json({"type": "error", "message": "No pending permission request to approve"})
        return

    await websocket.send_json(
        {
            "type": "permission_retry_started",
            "message": "Permission approved. Continuing current turn...",
        }
    )
    await stream_claude_turn(
        websocket,
        session,
        project_name=project_name,
        retry_from_permission=True,
    )


async def _handle_deny_permissions(
    websocket: WebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
) -> None:
    if not session.has_pending_permission_denials:
        await websocket.send_json({"type": "error", "message": "No pending permission request to deny"})
        return

    deny_message_raw = message.get("message", "Permission denied by user.")
    deny_message = (
        str(deny_message_raw).strip()
        if isinstance(deny_message_raw, str)
        else "Permission denied by user."
    )
    if not deny_message:
        deny_message = "Permission denied by user."

    await websocket.send_json(
        {
            "type": "permission_retry_started",
            "message": "Permission denied. Continuing current turn...",
        }
    )
    await stream_claude_turn(
        websocket,
        session,
        project_name=project_name,
        deny_from_permission_message=deny_message,
    )


async def _handle_abort_turn(
    websocket: WebSocket,
    session,
    project_name: str,
) -> None:
    """Handle abort request to stop the current turn."""
    try:
        aborted = await session.abort_current_turn()
        if aborted:
            await websocket.send_json({
                "type": "turn_aborted",
                "message": "Turn aborted by user",
            })
            print(f"[chat_ws] project={project_name} turn aborted")
        else:
            await websocket.send_json({
                "type": "error",
                "message": "No turn in progress to abort",
            })
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to abort turn: {str(e)}",
        })


async def _handle_firebase_auth_message(
    websocket: WebSocket,
    message: dict[str, Any],
    *,
    local_port: int,
) -> None:
    result = await process_firebase_auth_message_for_current_server(
        message,
        local_port=local_port,
    )
    await websocket.send_json(result.payload)
    if result.log_message:
        print(result.log_message)


async def _handle_disconnect_server_message(websocket: WebSocket) -> None:
    result = await process_disconnect_server_message_for_current_server()
    await websocket.send_json(result.payload)
    if result.log_message:
        print(result.log_message)


async def _dispatch_chat_message(
    websocket: WebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
    *,
    local_port: int,
) -> None:
    """Dispatch websocket chat message to corresponding handler."""
    message_type = message.get("type")
    print(f"[chat_ws] project={project_name} message_type={message_type}")
    handlers: dict[str, Callable[[], Awaitable[None]]] = {
        "message": lambda: _handle_user_message(websocket, session, project_name, message),
        "approve_permissions": lambda: _handle_approve_permissions(websocket, session, project_name),
        "deny_permissions": lambda: _handle_deny_permissions(websocket, session, project_name, message),
        "abort": lambda: _handle_abort_turn(websocket, session, project_name),
        "ping": lambda: websocket.send_json({"type": "pong"}),
        "firebase_auth": lambda: _handle_firebase_auth_message(
            websocket,
            message,
            local_port=local_port,
        ),
        "disconnect_server": lambda: _handle_disconnect_server_message(websocket),
    }

    handler = handlers.get(message_type)
    if handler is None:
        await websocket.send_json({"type": "error", "message": f"Unknown message type: {message_type}"})
        return
    await handler()


@router.websocket("/ws/chat/{project_name}")
@router.websocket("/ws/claude/{project_name}")  # Backwards compatibility alias
async def chat_websocket(
    websocket: WebSocket,
    project_name: str,
    api_key: Optional[str] = Query(None),
):
    """WebSocket endpoint for LLM chat communication.

    Includes connection limits and rate limiting for security.
    """
    # Check connection limits before accepting
    can_connect, reject_reason = _ws_manager.can_connect(websocket)
    if not can_connect:
        await websocket.close(code=1008, reason=reject_reason)
        print(f"[chat_ws] rejected connection: {reject_reason}")
        return

    access = validate_chat_websocket_access_for_current_server(api_key, project_name)
    if not access.success:
        if access.close_code is not None:
            await websocket.close(code=access.close_code, reason=access.close_reason or "")
            return

        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": access.error_message or "Project access denied",
            }
        )
        await websocket.close()
        return

    await websocket.accept()
    _ws_manager.register_connection(websocket)
    print(f"[chat_ws] accepted project={project_name} path={websocket.url.path}")

    project_path = access.project_path or ""
    local_port = access.local_port or 0

    selection_result = resolve_chat_provider_selection_for_current_server()
    if not selection_result.success or selection_result.selection is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": selection_result.error_message or "Failed to resolve provider",
            }
        )
        await websocket.close()
        return

    provider_name = selection_result.provider_name or "LLM"
    await websocket.send_json({"type": "status", "message": f"Connecting to {provider_name}..."})

    session_result = await create_chat_session_for_current_server(
        project_name,
        project_path,
        selection_result.selection,
    )
    if not session_result.success or session_result.session is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": session_result.error_message or "Failed to create chat session",
            }
        )
        await websocket.close()
        return

    session = session_result.session
    await websocket.send_json({"type": "status", "message": f"Connected to {provider_name}"})
    print(f"[chat_ws] project={project_name} connected provider={provider_name}")

    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Check message rate limit and re-resolve provider for user messages
                message_type = message.get("type")
                if message_type == "message":
                    # Rate limit check
                    can_send, rate_reason = _ws_manager.can_send_message(websocket)
                    if not can_send:
                        await websocket.send_json({
                            "type": "error",
                            "message": rate_reason,
                        })
                        continue

                    # Re-resolve provider for each new user turn so LLM selection changes
                    # take effect without requiring websocket reconnect.
                    latest_selection = resolve_chat_provider_selection_for_current_server()
                    if not latest_selection.success or latest_selection.selection is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": latest_selection.error_message
                                or "Failed to resolve provider",
                            }
                        )
                        continue

                    latest_provider_name = latest_selection.provider_name or "LLM"
                    latest_session_result = await create_chat_session_for_current_server(
                        project_name,
                        project_path,
                        latest_selection.selection,
                    )
                    if not latest_session_result.success or latest_session_result.session is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": latest_session_result.error_message
                                or "Failed to create chat session",
                            }
                        )
                        continue

                    session = latest_session_result.session
                    if latest_provider_name != provider_name:
                        provider_name = latest_provider_name
                        await websocket.send_json(
                            {
                                "type": "status",
                                "message": f"Switched to {provider_name}",
                            }
                        )
                        print(
                            f"[chat_ws] project={project_name} provider_switched={provider_name}"
                        )

                await _dispatch_chat_message(
                    websocket,
                    session,
                    project_name,
                    message,
                    local_port=local_port,
                )

            except WebSocketDisconnect:
                print(f"[chat_ws] project={project_name} disconnected")
                break
            except json.JSONDecodeError:
                print(f"[chat_ws] project={project_name} invalid_json")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
    finally:
        # Always unregister connection when done
        _ws_manager.unregister_connection(websocket)
