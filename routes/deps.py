"""Common dependencies for API routes."""

import asyncio
import logging
from typing import Any, Optional

from fastapi import Cookie, Header, HTTPException, Query, Request, WebSocket

from auth.auth_service import validate_api_key_for_current_server
from dashboard.dashboard_auth_service import get_dashboard_auth_status
from entitlement import EntitlementResult, get_entitlement_service

logger = logging.getLogger(__name__)

# Default interval for the WS periodic re-authentication loop (seconds).
# Picked to bound revocation latency to ~30s without producing meaningful
# load on the pairing/config check path.
WS_REAUTH_INTERVAL_SECONDS = 30.0

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


async def verify_subscription(
    request: Request,
    x_app_user_id: Optional[str] = Header(None, alias="X-App-User-Id"),
    x_entitlement_token: Optional[str] = Header(None, alias="X-Entitlement-Token"),
) -> EntitlementResult:
    """Reject callers without an active Code Bridge subscription.

    Code Bridge ships a single paid plan: every gated route requires either
    a local (loopback) caller, or an active RevenueCat ``code_bridge_pro``
    entitlement for the user identified by ``X-App-User-Id``.

    Local (loopback) requests bypass the check entirely — same convention
    as :func:`is_localhost_request`. This matters because the dashboard
    and the server-side maintenance flows run on ``127.0.0.1`` without an
    app user id.

    ``X-Entitlement-Token`` is reserved for a future signed-token path; it
    is accepted but not yet validated.
    """

    if is_localhost_request(request) and not is_request_from_tunnel(request):
        return EntitlementResult(
            active=True, reason="loopback", source="loopback"
        )

    # ``x_entitlement_token`` is reserved for future use; reference it so
    # linters don't flag the unused parameter and so callers can begin
    # sending it without a 400.
    _ = x_entitlement_token

    service = get_entitlement_service()
    result = service.is_entitled(x_app_user_id)
    if not result.active:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "subscription_required",
                "reason": result.reason,
            },
        )
    return result


async def _periodic_reauth_loop(
    websocket: Any,
    api_key: Optional[str],
    interval_seconds: float,
    *,
    validator: Any = None,
) -> None:
    """Background loop: revalidate ``api_key`` every ``interval_seconds``.

    On revocation (validator returns ``success=False``), closes the websocket
    with code 4001 / reason ``auth_invalid`` so clients (already hardened by
    BridgeService WS-2 phase 2 work) can drop into the auth-failed UI and
    re-authenticate. Transient validator exceptions are logged at WARNING and
    do **not** close the socket — only an explicit ``success=False`` from the
    validator is treated as revocation.
    """
    check = validator or validate_api_key_for_current_server
    while True:
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise

        try:
            result = check(api_key)
            still_valid = bool(getattr(result, "success", False))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "periodic reauth check raised; will retry next tick",
                exc_info=True,
            )
            continue

        if still_valid:
            continue

        logger.info("periodic reauth detected revoked api key; closing ws")
        try:
            await websocket.close(code=4001, reason="auth_invalid")
        except Exception:
            logger.debug("periodic reauth: close raised", exc_info=True)
        return


def start_periodic_reauth_task(
    websocket: Any,
    api_key: Optional[str],
    *,
    interval_seconds: float = WS_REAUTH_INTERVAL_SECONDS,
    validator: Any = None,
) -> asyncio.Task:
    """Spawn a background task that re-validates ``api_key`` on a fixed cadence.

    Callers must ``cancel()`` the returned task in their ``finally`` block to
    avoid leaks when the websocket disconnects.
    """
    return asyncio.create_task(
        _periodic_reauth_loop(
            websocket,
            api_key,
            interval_seconds,
            validator=validator,
        )
    )
