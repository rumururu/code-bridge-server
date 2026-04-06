"""WebSocket connection manager for chat service.

Provides connection limits and rate limiting for WebSocket connections.
"""

import time
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

# WebSocket connection limits
MAX_CONNECTIONS_PER_IP = 20  # Max concurrent connections per IP
MAX_MESSAGES_PER_MINUTE = 60  # Max messages per minute per connection
CONNECTION_RATE_LIMIT_WINDOW = 60  # Seconds
CONNECTION_RATE_LIMIT_MAX = 100  # Max connection attempts per window (effectively unlimited)


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


# Global connection manager instance
_ws_manager: WebSocketConnectionManager | None = None


def get_ws_manager() -> WebSocketConnectionManager:
    """Get the global WebSocket connection manager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketConnectionManager()
    return _ws_manager
