"""WebSocket chat routes for LLM streaming."""

import json
import logging
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

from chat.chat_stream_service import stream_claude_turn
from chat.chat_ws_service import (
    create_chat_session_for_current_server,
    process_disconnect_server_message_for_current_server,
    process_firebase_auth_message_for_current_server,
    resolve_chat_provider_selection_for_current_server,
    validate_chat_websocket_access_for_current_server,
)
from pairing.pairing import get_pairing_service
from services.chat.ws_manager import get_ws_manager

router = APIRouter(tags=["chat"])

# Get the global WebSocket connection manager
_ws_manager = get_ws_manager()


async def _handle_user_message(
    websocket: WebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
) -> None:
    user_message_raw = message.get("content", "")
    user_message = str(user_message_raw).strip()
    logger.info("project=%s incoming user_message len=%d", project_name, len(user_message))
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
            logger.info("project=%s turn aborted", project_name)
        else:
            await websocket.send_json({
                "type": "error",
                "message": "No turn in progress to abort",
            })
    except OSError as e:
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
        logger.info("%s", result.log_message)


async def _handle_disconnect_server_message(websocket: WebSocket) -> None:
    result = await process_disconnect_server_message_for_current_server()
    await websocket.send_json(result.payload)
    if result.log_message:
        logger.info("%s", result.log_message)


async def _handle_ping(websocket: WebSocket, api_key: Optional[str]) -> None:
    """Handle ping message and update client last_used timestamp."""
    if api_key:
        get_pairing_service().touch_api_key(api_key)
    await websocket.send_json({"type": "pong"})


async def _dispatch_chat_message(
    websocket: WebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
    *,
    local_port: int,
    api_key: Optional[str] = None,
) -> None:
    """Dispatch websocket chat message to corresponding handler."""
    message_type = message.get("type")
    logger.debug("project=%s message_type=%s", project_name, message_type)
    handlers: dict[str, Callable[[], Awaitable[None]]] = {
        "message": lambda: _handle_user_message(websocket, session, project_name, message),
        "approve_permissions": lambda: _handle_approve_permissions(websocket, session, project_name),
        "deny_permissions": lambda: _handle_deny_permissions(websocket, session, project_name, message),
        "abort": lambda: _handle_abort_turn(websocket, session, project_name),
        "ping": lambda: _handle_ping(websocket, api_key),
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
) -> None:
    """WebSocket endpoint for LLM chat communication.

    Includes connection limits and rate limiting for security.
    """
    # Check connection limits before accepting
    can_connect, reject_reason = _ws_manager.can_connect(websocket)
    if not can_connect:
        await websocket.close(code=1008, reason=reject_reason)
        logger.warning("rejected connection: %s", reject_reason)
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
    logger.info("accepted project=%s path=%s", project_name, websocket.url.path)

    try:
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

        logger.warning("[chat_ws] creating session project=%s path=%r", project_name, project_path)
        session_result = await create_chat_session_for_current_server(
            project_name,
            project_path,
            selection_result.selection,
        )
        logger.warning("[chat_ws] session create result success=%s error=%s", session_result.success, session_result.error_message)
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
        logger.info("project=%s connected provider=%s", project_name, provider_name)
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
                        logger.info(
                            "project=%s provider_switched=%s", project_name, provider_name
                        )

                await _dispatch_chat_message(
                    websocket,
                    session,
                    project_name,
                    message,
                    local_port=local_port,
                    api_key=api_key,
                )

            except WebSocketDisconnect:
                logger.info("project=%s disconnected", project_name)
                break
            except json.JSONDecodeError:
                logger.warning("project=%s invalid_json", project_name)
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
    finally:
        # Always unregister connection when done
        _ws_manager.unregister_connection(websocket)
