"""Shared helpers for Firebase authentication and remote device registration."""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional

from core.config import get_config
from system.optional_services import (
    FIREBASE_AVAILABLE,
    TUNNEL_AVAILABLE,
    create_tunnel_service,
    get_firebase_auth,
    get_tunnel_service,
)
from pairing.pairing import get_pairing_service
from .remote_access_models import (
    PairingRemoteAccessResult,
    PairVerifyFlowResult,
    RemoteAccessActionResult,
    RemoteAccessLoginPayload,
    RemoteAccessLoginResult,
    RemoteFirebaseStatus,
    RemoteMdnsStatus,
    RemoteNetworkStatus,
    RemoteTunnelStatus,
)

logger = logging.getLogger(__name__)


def _resolve_firebase_auth(firebase_auth: Any | None) -> Any | None:
    return firebase_auth or get_firebase_auth()


def _firebase_unavailable_login_result() -> RemoteAccessLoginResult:
    return RemoteAccessLoginResult(
        success=False,
        status_code=400,
        error="Firebase is not available",
    )


def _firebase_unavailable_action_result() -> RemoteAccessActionResult:
    return RemoteAccessActionResult(
        success=False,
        status_code=400,
        error="Firebase is not available",
    )


def parse_remote_access_login_payload(body: Any) -> tuple[RemoteAccessLoginPayload | None, str | None]:
    """Validate and normalize remote access login body."""
    if not isinstance(body, dict):
        return None, "Invalid request body. Expected JSON with 'id_token' field."

    raw_id_token = body.get("id_token")
    if not isinstance(raw_id_token, str) or not raw_id_token.strip():
        return None, "Missing 'id_token' in request body"
    id_token = raw_id_token.strip()

    raw_refresh_token = body.get("refresh_token")
    refresh_token = raw_refresh_token if isinstance(raw_refresh_token, str) else None

    register_device = bool(body.get("register_device", True))

    return (
        RemoteAccessLoginPayload(
            id_token=id_token,
            refresh_token=refresh_token,
            register_device=register_device,
        ),
        None,
    )


async def _resolve_tunnel_url_for_registration(*, local_port: int, autostart_tunnel: bool) -> Optional[str]:
    if not TUNNEL_AVAILABLE:
        return None

    tunnel_service = get_tunnel_service()
    if tunnel_service:
        if autostart_tunnel and not tunnel_service.is_running:
            logger.info("Auto-starting tunnel for remote access...")
            return await tunnel_service.start()
        return tunnel_service.tunnel_url

    if autostart_tunnel:
        tunnel_service = create_tunnel_service(local_port=local_port)
        if tunnel_service:
            logger.info("Creating and starting tunnel for remote access...")
            return await tunnel_service.start()

    return None


async def register_device_for_remote_access(
    firebase_auth: Any,
    *,
    local_port: int,
    autostart_tunnel: bool,
) -> bool:
    """Register the current server device for remote access."""
    tunnel_url = await _resolve_tunnel_url_for_registration(
        local_port=local_port,
        autostart_tunnel=autostart_tunnel,
    )

    pairing = get_pairing_service()
    local_url = f"http://{pairing.get_local_ip()}:{local_port}"
    return await firebase_auth.register_device(tunnel_url, local_url)


async def check_ownership_conflict(
    firebase_id_token: str,
) -> tuple[bool, Optional[str]]:
    """Check if there's an ownership conflict without consuming any resources.

    Returns:
        Tuple of (has_conflict, current_owner_email)
    """
    if not firebase_id_token or not FIREBASE_AVAILABLE:
        return False, None

    firebase_auth = get_firebase_auth()
    if not firebase_auth:
        return False, None

    # Check if Firebase is initialized
    if not firebase_auth.is_initialized:
        # Try to initialize Firebase
        try:
            await firebase_auth.initialize()
        except Exception:
            # If initialization fails, skip ownership check
            return False, None

    # Verify the incoming token to get new user's info (without storing)
    token_info = await firebase_auth.verify_id_token(firebase_id_token)
    if not token_info:
        return False, None

    new_user_id = token_info.get("user_id")
    current_owner_id = firebase_auth.user_id
    current_owner_email = firebase_auth.email

    # If there's a different owner, return conflict
    if current_owner_id and new_user_id and current_owner_id != new_user_id:
        logger.info(
            "Ownership conflict detected: current=%s, new=%s",
            current_owner_email or current_owner_id,
            token_info.get("email") or new_user_id,
        )
        return True, current_owner_email

    return False, None


async def register_pairing_remote_access(
    *,
    firebase_id_token: str,
    firebase_refresh_token: Optional[str],
    local_port: int,
    force_replace: bool = False,
) -> PairingRemoteAccessResult:
    """Authenticate Firebase token and register device during pairing.

    Args:
        firebase_id_token: Firebase ID token from the app
        firebase_refresh_token: Firebase refresh token (optional)
        local_port: Local API server port
        force_replace: If True, skip ownership conflict check and replace existing owner

    Returns:
        PairingRemoteAccessResult with ownership_conflict=True if server
        is already paired with a different account and force_replace is False
    """
    if not firebase_id_token or not FIREBASE_AVAILABLE:
        return PairingRemoteAccessResult()

    firebase_auth = get_firebase_auth()
    if not firebase_auth:
        return PairingRemoteAccessResult()

    # Check for ownership conflict before authenticating
    if not force_replace:
        # Check if Firebase is initialized before verifying token
        if not firebase_auth.is_initialized:
            try:
                await firebase_auth.initialize()
            except Exception:
                # If initialization fails, skip ownership check and proceed
                pass

        # Verify the incoming token to get new user's info (without storing)
        if firebase_auth.is_initialized:
            token_info = await firebase_auth.verify_id_token(firebase_id_token)
        else:
            token_info = None

        if token_info:
            new_user_id = token_info.get("user_id")
            current_owner_id = firebase_auth.user_id
            current_owner_email = firebase_auth.email

            # If there's a different owner, return conflict
            if current_owner_id and new_user_id and current_owner_id != new_user_id:
                logger.info(
                    "Ownership conflict: current=%s, new=%s",
                    current_owner_email or current_owner_id,
                    token_info.get("email") or new_user_id,
                )
                return PairingRemoteAccessResult(
                    ownership_conflict=True,
                    current_owner_email=current_owner_email,
                )

    auth_success = await firebase_auth.authenticate_with_token(
        firebase_id_token,
        refresh_token=firebase_refresh_token,
    )
    if not auth_success:
        return PairingRemoteAccessResult(
            firebase_registered=False,
            firebase_error="Token verification failed",
        )

    # Add to paired accounts for multi-account support
    try:
        from firebase.paired_accounts import get_paired_accounts_manager
        status = firebase_auth.get_status()
        user_id = status.get("user_id")
        email = status.get("email")
        if user_id:
            paired_accounts = get_paired_accounts_manager()
            paired_accounts.add_or_update_account(
                user_id=user_id,
                email=email,
                id_token=firebase_id_token,
                refresh_token=firebase_refresh_token,
            )
            logger.info("Added primary owner to paired accounts: %s", email or user_id)
    except Exception as exc:
        logger.warning("Failed to add to paired accounts: %s", exc)

    registered = await register_device_for_remote_access(
        firebase_auth,
        local_port=local_port,
        autostart_tunnel=True,
    )
    if registered:
        logger.info("Server registered to Firebase for user: %s", firebase_auth.get_status().get('user_id'))

    return PairingRemoteAccessResult(firebase_registered=registered)


async def login_for_remote_access(
    payload: RemoteAccessLoginPayload,
    *,
    local_port: int,
    firebase_available: bool = FIREBASE_AVAILABLE,
    firebase_auth: Any | None = None,
) -> RemoteAccessLoginResult:
    """Authenticate Firebase login payload and optionally register the device."""
    if not firebase_available:
        return _firebase_unavailable_login_result()

    resolved_firebase_auth = _resolve_firebase_auth(firebase_auth)
    if not resolved_firebase_auth:
        return RemoteAccessLoginResult(
            success=False,
            status_code=500,
            error="Firebase auth not initialized",
        )

    success = await resolved_firebase_auth.authenticate_with_token(
        payload.id_token,
        refresh_token=payload.refresh_token,
    )
    if not success:
        return RemoteAccessLoginResult(
            success=False,
            status_code=401,
            error="Token verification failed",
        )

    status = resolved_firebase_auth.get_status()

    # Add to paired accounts for multi-account support
    try:
        from firebase.paired_accounts import get_paired_accounts_manager
        user_id = status.get("user_id")
        email = status.get("email")
        if user_id:
            paired_accounts = get_paired_accounts_manager()
            paired_accounts.add_or_update_account(
                user_id=user_id,
                email=email,
                id_token=payload.id_token,
                refresh_token=payload.refresh_token,
            )
    except Exception as exc:
        logger.warning("Failed to add to paired accounts: %s", exc)

    if payload.register_device:
        await register_device_for_remote_access(
            resolved_firebase_auth,
            local_port=local_port,
            autostart_tunnel=False,
        )

    raw_user_id = status.get("user_id")
    user_id = raw_user_id if isinstance(raw_user_id, str) else None
    raw_server_id = status.get("server_id")
    server_id = raw_server_id if isinstance(raw_server_id, str) else None
    raw_server_name = status.get("server_name")
    server_name = raw_server_name if isinstance(raw_server_name, str) else None

    return RemoteAccessLoginResult(
        success=True,
        status_code=200,
        user_id=user_id,
        server_id=server_id,
        server_name=server_name,
    )


async def login_for_remote_access_for_current_server(
    payload: RemoteAccessLoginPayload,
    *,
    firebase_available: bool = FIREBASE_AVAILABLE,
    firebase_auth: Any | None = None,
) -> RemoteAccessLoginResult:
    """Authenticate remote access using current server config."""
    config = get_config()
    return await login_for_remote_access(
        payload,
        local_port=config.api_port,
        firebase_available=firebase_available,
        firebase_auth=firebase_auth,
    )


async def login_for_remote_access_body_for_current_server(body: Any) -> RemoteAccessLoginResult:
    """Parse login body and authenticate remote access using current server config."""
    payload, parse_error = parse_remote_access_login_payload(body)
    if payload is None:
        return RemoteAccessLoginResult(
            success=False,
            status_code=400,
            error=parse_error or "Invalid request body",
        )

    return await login_for_remote_access_for_current_server(
        payload,
        firebase_available=FIREBASE_AVAILABLE,
        firebase_auth=get_firebase_auth(),
    )


async def login_for_remote_access_request_json_for_current_server(
    request_json_loader: Callable[[], Awaitable[Any]],
) -> RemoteAccessLoginResult:
    """Read request JSON and process remote-access login for current server."""
    try:
        body = await request_json_loader()
    except (ValueError, json.JSONDecodeError):
        return RemoteAccessLoginResult(
            success=False,
            status_code=400,
            error="Invalid request body. Expected JSON with 'id_token' field.",
        )

    return await login_for_remote_access_body_for_current_server(body)


async def verify_pairing_flow(
    *,
    pairing_service: Any,
    pair_token: str,
    client_id: Optional[str],
    device_name: Optional[str],
    firebase_id_token: Optional[str],
    firebase_refresh_token: Optional[str],
    local_port: int,
    force_replace: bool = False,
) -> PairVerifyFlowResult:
    """Verify pair token and optionally register Firebase remote access.

    Args:
        force_replace: If True, replace existing server owner without confirmation
    """
    # IMPORTANT: Check ownership conflict BEFORE consuming the pair token
    # This prevents the token from being marked "used" when there's a conflict
    if firebase_id_token and not force_replace:
        has_conflict, current_owner_email = await check_ownership_conflict(
            firebase_id_token
        )
        if has_conflict:
            # Return conflict info WITHOUT consuming the pair token
            # User can retry with the same QR code after resolving conflict
            return PairVerifyFlowResult(
                success=True,
                status_code=200,
                api_key=None,  # No API key yet - token not consumed
                server_id=pairing_service.server_id,
                client_id=None,
                ownership_conflict=True,
                current_owner_email=current_owner_email,
            )

    # Now safe to consume the pair token
    verify_result = pairing_service.verify_pair_token(
        pair_token=pair_token,
        client_id=client_id,
        device_name=device_name,
    )

    if not verify_result.success:
        error_message = verify_result.error or "Pairing failed"
        return PairVerifyFlowResult(
            success=False,
            status_code=verify_result.status_code,
            error=error_message,
        )

    firebase_registered: Optional[bool] = None
    firebase_error: Optional[str] = None

    if firebase_id_token:
        # Ownership already checked above, now register with force_replace=True
        # to skip the redundant check in register_pairing_remote_access
        remote_result = await register_pairing_remote_access(
            firebase_id_token=firebase_id_token,
            firebase_refresh_token=firebase_refresh_token,
            local_port=local_port,
            force_replace=True,  # Already checked above
        )
        firebase_registered = remote_result.firebase_registered
        firebase_error = remote_result.firebase_error

        # Ownership conflict should not happen here since we checked above
        if remote_result.ownership_conflict:
            logger.warning("Unexpected ownership conflict after pre-check")
            return PairVerifyFlowResult(
                success=True,
                status_code=200,
                api_key=verify_result.api_key,
                server_id=verify_result.server_id,
                client_id=verify_result.client_id,
                ownership_conflict=True,
                current_owner_email=remote_result.current_owner_email,
            )

        # Update paired client with Firebase user info
        if firebase_registered and FIREBASE_AVAILABLE:
            firebase_auth = get_firebase_auth()
            if firebase_auth:
                status = firebase_auth.get_status()
                firebase_user_id = status.get("user_id")
                firebase_email = status.get("email")
                if firebase_user_id or firebase_email:
                    pairing_service.update_client_firebase_user(
                        verify_result.client_id,
                        firebase_user_id=firebase_user_id,
                        firebase_email=firebase_email,
                    )

    return PairVerifyFlowResult(
        success=True,
        status_code=200,
        api_key=verify_result.api_key,
        server_id=verify_result.server_id,
        client_id=verify_result.client_id,
        firebase_registered=firebase_registered,
        firebase_error=firebase_error,
    )


async def verify_pairing_flow_for_current_server(
    *,
    pairing_service: Any,
    pair_token: str,
    client_id: Optional[str],
    device_name: Optional[str],
    firebase_id_token: Optional[str],
    firebase_refresh_token: Optional[str],
    force_replace: bool = False,
) -> PairVerifyFlowResult:
    """Verify pair token flow using the configured local server port."""
    config = get_config()
    return await verify_pairing_flow(
        pairing_service=pairing_service,
        pair_token=pair_token,
        client_id=client_id,
        device_name=device_name,
        firebase_id_token=firebase_id_token,
        firebase_refresh_token=firebase_refresh_token,
        local_port=config.api_port,
        force_replace=force_replace,
    )


async def verify_pair_token_for_current_server(
    *,
    pair_token: str,
    client_id: Optional[str],
    device_name: Optional[str],
    firebase_id_token: Optional[str],
    firebase_refresh_token: Optional[str],
    force_replace: bool = False,
    pairing_service: Any | None = None,
) -> PairVerifyFlowResult:
    """Verify pair token using current pairing and server config context."""
    resolved_pairing_service = pairing_service or get_pairing_service()
    return await verify_pairing_flow_for_current_server(
        pairing_service=resolved_pairing_service,
        pair_token=pair_token,
        client_id=client_id,
        device_name=device_name,
        firebase_id_token=firebase_id_token,
        firebase_refresh_token=firebase_refresh_token,
        force_replace=force_replace,
    )


async def logout_remote_access(
    *,
    firebase_available: bool = FIREBASE_AVAILABLE,
    firebase_auth: Any | None = None,
) -> RemoteAccessActionResult:
    """Sign out remote-access Firebase session."""
    if not firebase_available:
        return _firebase_unavailable_action_result()

    resolved_firebase_auth = _resolve_firebase_auth(firebase_auth)
    if resolved_firebase_auth:
        await resolved_firebase_auth.sign_out()

    return RemoteAccessActionResult(success=True, status_code=200)


async def logout_remote_access_for_current_server() -> RemoteAccessActionResult:
    """Sign out from remote access using current optional-service context."""
    return await logout_remote_access(
        firebase_available=FIREBASE_AVAILABLE,
        firebase_auth=get_firebase_auth(),
    )


async def disconnect_remote_access(
    *,
    firebase_available: bool = FIREBASE_AVAILABLE,
    firebase_auth: Any | None = None,
) -> RemoteAccessActionResult:
    """Completely clear remote-access Firebase session."""
    if not firebase_available:
        return _firebase_unavailable_action_result()

    resolved_firebase_auth = _resolve_firebase_auth(firebase_auth)
    if resolved_firebase_auth:
        await resolved_firebase_auth.clear_auth()

    return RemoteAccessActionResult(
        success=True,
        status_code=200,
        message="Disconnected from remote access",
    )


async def disconnect_remote_access_for_current_server() -> RemoteAccessActionResult:
    """Disconnect remote access using current optional-service context."""
    return await disconnect_remote_access(
        firebase_available=FIREBASE_AVAILABLE,
        firebase_auth=get_firebase_auth(),
    )


def build_remote_network_status(config: Any) -> RemoteNetworkStatus:
    """Build typed network/remote-access status payload."""
    tunnel_running, tunnel_url, tunnel_installed = _extract_tunnel_status()
    firebase_authenticated, firebase_user_id, firebase_server_id = _extract_firebase_status()

    return RemoteNetworkStatus(
        mdns=RemoteMdnsStatus(
            available=False,
            enabled=False,
            registered=False,
            server_name=config.server_name,
        ),
        tunnel=RemoteTunnelStatus(
            available=TUNNEL_AVAILABLE,
            enabled=config.remote_access_enabled,
            running=tunnel_running,
            url=tunnel_url,
            installed=tunnel_installed,
        ),
        firebase=RemoteFirebaseStatus(
            available=FIREBASE_AVAILABLE,
            enabled=FIREBASE_AVAILABLE,  # Enabled if available (no separate config)
            authenticated=firebase_authenticated,
            user_id=firebase_user_id,
            server_id=firebase_server_id,
        ),
    )


def _extract_tunnel_status() -> tuple[bool, Optional[str], Optional[bool]]:
    tunnel_running = False
    tunnel_url: Optional[str] = None
    tunnel_installed: Optional[bool] = None

    tunnel_service = get_tunnel_service()
    if tunnel_service:
        tunnel_status = tunnel_service.get_status()
        tunnel_running = bool(tunnel_status.get("running", False))
        raw_url = tunnel_status.get("url")
        tunnel_url = raw_url if isinstance(raw_url, str) else None
        raw_installed = tunnel_status.get("installed")
        if isinstance(raw_installed, bool):
            tunnel_installed = raw_installed

    return tunnel_running, tunnel_url, tunnel_installed


def _extract_firebase_status() -> tuple[bool, Optional[str], Optional[str]]:
    firebase_authenticated = False
    firebase_user_id: Optional[str] = None
    firebase_server_id: Optional[str] = None

    if FIREBASE_AVAILABLE:
        firebase_auth = get_firebase_auth()
        if firebase_auth:
            auth_status = firebase_auth.get_status()
            firebase_authenticated = bool(auth_status.get("authenticated", False))
            raw_user_id = auth_status.get("user_id")
            firebase_user_id = raw_user_id if isinstance(raw_user_id, str) else None
            raw_server_id = auth_status.get("server_id")
            firebase_server_id = raw_server_id if isinstance(raw_server_id, str) else None

    return firebase_authenticated, firebase_user_id, firebase_server_id


def build_remote_network_status_for_current_server() -> RemoteNetworkStatus:
    """Build remote-network status using current server config."""
    config = get_config()
    return build_remote_network_status(config)


async def start_tunnel_for_remote_access(*, local_port: int) -> RemoteAccessActionResult:
    """Start tunnel service for remote access."""
    if not TUNNEL_AVAILABLE:
        return RemoteAccessActionResult(
            success=False,
            status_code=400,
            error="Tunnel service not available",
        )

    tunnel_service = get_tunnel_service()
    if not tunnel_service:
        tunnel_service = create_tunnel_service(local_port=local_port)

    if not tunnel_service:
        return RemoteAccessActionResult(
            success=False,
            status_code=500,
            error="Failed to initialize tunnel service",
        )

    url = await tunnel_service.start()
    if url:
        # Register tunnel URL to Firebase for remote discovery
        if FIREBASE_AVAILABLE:
            firebase_auth = get_firebase_auth()
            if firebase_auth and firebase_auth.get_status().get("authenticated"):
                pairing = get_pairing_service()
                local_url = f"http://{pairing.get_local_ip()}:{local_port}"
                registered = await firebase_auth.register_device(url, local_url)
                if registered:
                    logger.info("Tunnel URL registered to Firebase: %s", url)

        return RemoteAccessActionResult(
            success=True,
            status_code=200,
            url=url,
        )

    return RemoteAccessActionResult(
        success=False,
        status_code=500,
        error="Failed to start tunnel",
    )


async def start_tunnel_for_current_server() -> RemoteAccessActionResult:
    """Start tunnel using configured local server port.

    Rejects tunnel start if IP Login is enabled, as external access
    should not be allowed when authentication is disabled.
    """
    from system.system_settings_service import get_allow_ip_login

    if get_allow_ip_login():
        return RemoteAccessActionResult(
            success=False,
            status_code=400,
            error="Cannot start tunnel while IP Login is enabled. "
            "Disable IP Login first for security.",
        )

    config = get_config()
    return await start_tunnel_for_remote_access(local_port=config.api_port)


async def stop_tunnel_for_remote_access() -> RemoteAccessActionResult:
    """Stop tunnel service for remote access."""
    tunnel_service = get_tunnel_service()
    if tunnel_service:
        await tunnel_service.stop()
        return RemoteAccessActionResult(success=True, status_code=200)
    return RemoteAccessActionResult(
        success=True,
        status_code=200,
        message="No tunnel running",
    )


async def stop_tunnel_for_current_server() -> RemoteAccessActionResult:
    """Stop tunnel for current server context."""
    return await stop_tunnel_for_remote_access()
