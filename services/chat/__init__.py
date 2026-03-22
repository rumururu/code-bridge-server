"""Chat services package.

Provides WebSocket management, streaming, and session management.
"""

from chat_stream_service import stream_claude_turn
from chat_session_service import (
    ChatProviderSelection,
    ChatSessionInitError,
    get_chat_provider_selection,
    create_chat_session,
)
from chat_ws_service import (
    create_chat_session_for_current_server,
    process_disconnect_server_message_for_current_server,
    process_firebase_auth_message_for_current_server,
    resolve_chat_provider_selection_for_current_server,
    validate_chat_websocket_access_for_current_server,
)
from services.chat.ws_manager import WebSocketConnectionManager, get_ws_manager

__all__ = [
    "stream_claude_turn",
    "ChatProviderSelection",
    "ChatSessionInitError",
    "get_chat_provider_selection",
    "create_chat_session",
    "create_chat_session_for_current_server",
    "process_disconnect_server_message_for_current_server",
    "process_firebase_auth_message_for_current_server",
    "resolve_chat_provider_selection_for_current_server",
    "validate_chat_websocket_access_for_current_server",
    "WebSocketConnectionManager",
    "get_ws_manager",
]
