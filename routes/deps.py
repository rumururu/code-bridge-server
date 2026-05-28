"""Common dependencies for API routes."""

from typing import Any, Optional

from fastapi import Cookie, Header, HTTPException, Query, Request, WebSocket

from auth.auth_service import validate_api_key_for_current_server
from dashboard.dashboard_auth_service import get_dashboard_auth_status

TUNNEL_HEADER_NAMES = (
    "CF-Connecting-IP",
    "CF-Ray",
    "CF-IPCountry",
    "CF-Visitor",
)


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None, alias="api_key"),
) -> Optional[str]:
    """Verify API key from header or query parameter.

    When accessed via tunnel (external), API key is ALWAYS required.
    IP login and legacy mode only grant anonymous access for LOCAL requests.
    """
    provided_key = x_api_key or api_key
    validation = validate_api_key_for_current_server(provided_key)

    # Tunnel access ALWAYS requires API key (pairing), regardless of IP login setting
    # IP login only grants anonymous access for local network, not external tunnel
    if validation.success and not provided_key and is_request_from_tunnel(request):
        raise HTTPException(
            status_code=401,
            detail="API key required for external access. "
            "Please pair your device using QR code.",
        )

    if not validation.success:
        raise HTTPException(status_code=401, detail=validation.error or "Invalid API key")

    return validation.api_key


def has_tunnel_headers(headers: Any) -> bool:
    """Return True when request headers indicate Cloudflare Tunnel access."""
    for header in TUNNEL_HEADER_NAMES:
        if headers.get(header):
            return True
    return False


def is_request_from_tunnel(request: Request) -> bool:
    """Check if request came through Cloudflare Tunnel.

    Cloudflare adds specific headers when proxying requests:
    - CF-Connecting-IP: Client's real IP
    - CF-Ray: Unique Cloudflare request ID
    - CF-IPCountry: Client's country code

    If any of these headers are present, the request came through the tunnel.
    """
    return has_tunnel_headers(request.headers)


def is_websocket_from_tunnel(websocket: WebSocket) -> bool:
    """Check if a WebSocket handshake came through Cloudflare Tunnel."""
    return has_tunnel_headers(websocket.headers)


async def require_local_access(request: Request) -> None:
    """Dependency that blocks external (tunnel) access.

    Use this for sensitive pages like /dashboard and /pair that should
    only be accessible from local network, not through tunnel.

    Raises 403 Forbidden if request came through Cloudflare Tunnel.
    """
    if is_request_from_tunnel(request):
        raise HTTPException(
            status_code=403,
            detail="This page is only accessible from local network. "
            "External access via tunnel is not allowed.",
        )


async def require_localhost_only(request: Request) -> None:
    """Dependency that requires request from localhost (127.0.0.1 or ::1).

    Use this for sensitive pages like /pair that should ONLY be accessible
    from the same machine running the server.

    Raises 403 Forbidden if request is not from localhost.
    """
    if is_request_from_tunnel(request):
        raise HTTPException(
            status_code=403,
            detail="This page is only accessible from localhost.",
        )
    if not is_localhost_request(request):
        raise HTTPException(
            status_code=403,
            detail="This page is only accessible from localhost (127.0.0.1). "
            "Open this URL on the same machine where the server is running.",
        )


def is_localhost_request(request: Request) -> bool:
    """Check if request is from localhost (127.0.0.1 or ::1)."""
    client = request.client
    if not client:
        return False
    host = client.host
    return host in ("127.0.0.1", "::1", "localhost")


async def verify_api_key_or_localhost(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None, alias="api_key"),
) -> Optional[str]:
    """Allow access if localhost OR valid API key.

    Used for settings endpoints that dashboard accesses locally.
    """
    # Allow localhost access without API key
    if is_localhost_request(request) and not is_request_from_tunnel(request):
        return None

    # Otherwise, require API key
    return await verify_api_key(request, x_api_key, api_key)


async def require_dashboard_auth(
    request: Request,
    dashboard_session: Optional[str] = Cookie(None),
) -> None:
    """Dependency that requires dashboard authentication.

    If password is enabled and user is not authenticated, raises 401.
    Should be used with require_local_access for full protection.
    """
    # First check local access
    if is_request_from_tunnel(request):
        raise HTTPException(
            status_code=403,
            detail="This page is only accessible from local network.",
        )

    # Check dashboard authentication
    status = get_dashboard_auth_status(dashboard_session)

    if status.password_enabled and not status.session_valid:
        raise HTTPException(
            status_code=401,
            detail="Dashboard authentication required",
        )
