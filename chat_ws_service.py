"""Service helpers for websocket chat route access and auth flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from chat_session_service import (
    ChatSessionInitError,
    create_chat_session,
    get_chat_provider_selection,
)
from config import get_config
from optional_services import get_firebase_auth
from pairing import get_pairing_service
from remote_access_service import (
    parse_remote_access_login_payload,
    register_device_for_remote_access,
)


@dataclass(frozen=True)
class ChatWebSocketAccessResult:
    """Access validation result for websocket chat endpoints."""

    success: bool
    close_code: int | None = None
    close_reason: str | None = None
    error_message: str | None = None
    project_path: str | None = None
    local_port: int | None = None


@dataclass(frozen=True)
class ChatProviderSelectionResult:
    """Provider selection result for websocket session setup."""

    success: bool
    provider_name: str | None = None
    selection: Any | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ChatSessionCreationResult:
    """Chat session creation result."""

    success: bool
    session: Any | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ChatWebSocketMessageResult:
    """Message payload to send to websocket clients."""

    payload: dict[str, str]
    log_message: str | None = None


def validate_chat_websocket_access_for_current_server(
    api_key: str | None,
    project_name: str,
    *,
    pairing_service: Any | None = None,
    config: Any | None = None,
) -> ChatWebSocketAccessResult:
    """Validate API key and project existence for websocket chat connection."""
    resolved_config = config or get_config()
    configured_api_key = str(getattr(resolved_config, "api_key", "") or "").strip()

    # Local/dev mode: static API key is unset, so websocket access is open.
    if configured_api_key:
        if api_key == configured_api_key:
            pass
        else:
            if not api_key:
                return ChatWebSocketAccessResult(
                    success=False,
                    close_code=4001,
                    close_reason="API key required",
                )

            resolved_pairing = pairing_service or get_pairing_service()
            if not resolved_pairing.validate_api_key(api_key):
                return ChatWebSocketAccessResult(
                    success=False,
                    close_code=4001,
                    close_reason="Invalid API key",
                )

    project = resolved_config.get_project(project_name)
    if project is None:
        return ChatWebSocketAccessResult(
            success=False,
            error_message=f"Project {project_name} not found",
        )

    project_path = project.get("path") if isinstance(project, dict) else None
    if not isinstance(project_path, str) or not project_path:
        return ChatWebSocketAccessResult(
            success=False,
            error_message=f"Project {project_name} has no valid path",
        )

    return ChatWebSocketAccessResult(
        success=True,
        project_path=project_path,
        local_port=int(getattr(resolved_config, "port", 0) or 0),
    )


def resolve_chat_provider_selection_for_current_server(
    *,
    selection_resolver: Callable[[], Any] | None = None,
) -> ChatProviderSelectionResult:
    """Resolve current chat provider selection with error normalization."""
    resolver = selection_resolver or get_chat_provider_selection
    try:
        selection = resolver()
    except ChatSessionInitError as exc:
        return ChatProviderSelectionResult(success=False, error_message=str(exc))

    provider_name = getattr(selection, "provider_name", None)
    if not isinstance(provider_name, str) or not provider_name:
        return ChatProviderSelectionResult(success=False, error_message="Invalid provider selection")

    return ChatProviderSelectionResult(
        success=True,
        provider_name=provider_name,
        selection=selection,
    )


async def create_chat_session_for_current_server(
    project_name: str,
    project_path: str,
    selection: Any,
    *,
    session_creator: Callable[..., Any] | None = None,
) -> ChatSessionCreationResult:
    """Create chat session and normalize setup errors."""
    creator = session_creator or create_chat_session
    try:
        session = await creator(
            project_name=project_name,
            project_path=project_path,
            selection=selection,
        )
    except ChatSessionInitError as exc:
        return ChatSessionCreationResult(success=False, error_message=str(exc))

    return ChatSessionCreationResult(success=True, session=session)


async def process_firebase_auth_message_for_current_server(
    message: dict[str, Any],
    *,
    local_port: int,
    payload_parser: Callable[[Any], tuple[Any | None, str | None]] | None = None,
    firebase_auth: Any | None = None,
    firebase_auth_factory: Callable[[], Any | None] | None = None,
    register_device_for_remote_access_fn: Callable[..., Any] | None = None,
) -> ChatWebSocketMessageResult:
    """Process firebase_auth websocket message and return response payload."""
    parser = payload_parser or parse_remote_access_login_payload
    payload, parse_error = parser(message)
    if payload is None:
        return ChatWebSocketMessageResult(
            payload={"type": "error", "message": parse_error or "Invalid payload"}
        )

    resolved_auth = firebase_auth
    if resolved_auth is None:
        auth_factory = firebase_auth_factory or get_firebase_auth
        resolved_auth = auth_factory()

    if not resolved_auth:
        return ChatWebSocketMessageResult(
            payload={"type": "error", "message": "Firebase not configured"}
        )

    if not await resolved_auth.initialize():
        return ChatWebSocketMessageResult(
            payload={"type": "error", "message": "Firebase initialization failed"}
        )

    success = await resolved_auth.authenticate_with_token(
        payload.id_token,
        refresh_token=payload.refresh_token,
        auth_mode=payload.auth_mode,
    )
    if not success:
        return ChatWebSocketMessageResult(
            payload={"type": "error", "message": "Firebase auth failed"},
            log_message="WebSocket: Firebase authentication failed",
        )

    register_fn = register_device_for_remote_access_fn or register_device_for_remote_access
    if payload.register_device:
        await register_fn(
            resolved_auth,
            local_port=local_port,
            autostart_tunnel=False,
        )

    return ChatWebSocketMessageResult(
        payload={
            "type": "status",
            "message": f"Firebase authenticated ({payload.auth_mode} mode)",
        },
        log_message=f"WebSocket: Firebase authentication successful ({payload.auth_mode} mode)",
    )


async def process_disconnect_server_message_for_current_server(
    *,
    firebase_auth: Any | None = None,
    firebase_auth_factory: Callable[[], Any | None] | None = None,
) -> ChatWebSocketMessageResult:
    """Process disconnect_server websocket message and clear auth state."""
    resolved_auth = firebase_auth
    if resolved_auth is None:
        auth_factory = firebase_auth_factory or get_firebase_auth
        resolved_auth = auth_factory()

    if resolved_auth:
        await resolved_auth.clear_auth()
        return ChatWebSocketMessageResult(
            payload={"type": "status", "message": "Server disconnected"},
            log_message="WebSocket: Server disconnected, auth cleared",
        )

    return ChatWebSocketMessageResult(
        payload={"type": "status", "message": "No auth to clear"}
    )
