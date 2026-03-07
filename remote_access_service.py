"""Shared helpers for Firebase authentication and remote device registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from config import get_config
from optional_services import (
    FIREBASE_AVAILABLE,
    TUNNEL_AVAILABLE,
    create_tunnel_service,
    get_firebase_auth,
    get_tunnel_service,
)
from pairing import get_pairing_service


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


@dataclass(frozen=True)
class RemoteAccessLoginPayload:
    """Parsed payload for remote access login requests."""

    id_token: str
    refresh_token: Optional[str]
    auth_mode: str
    register_device: bool


@dataclass(frozen=True)
class PairingRemoteAccessResult:
    """Typed result for pairing-time remote access registration."""

    firebase_registered: Optional[bool] = None
    firebase_error: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        """Convert to route response fields, omitting unset values."""
        payload: dict[str, Any] = {}
        if self.firebase_registered is not None:
            payload["firebase_registered"] = self.firebase_registered
        if self.firebase_error:
            payload["firebase_error"] = self.firebase_error
        return payload


@dataclass(frozen=True)
class ServiceFlowResult:
    """Common base for service flow results."""

    success: bool
    status_code: int
    error: Optional[str] = None

    def error_response(self, fallback_message: str) -> dict[str, Any]:
        return {"error": self.error or fallback_message}


@dataclass(frozen=True)
class RemoteAccessLoginResult(ServiceFlowResult):
    """Typed result for remote-access login flow."""

    user_id: Optional[str] = None
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    auth_mode: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("Remote access login failed")
        return {
            "success": True,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "auth_mode": self.auth_mode,
        }


@dataclass(frozen=True)
class PairVerifyFlowResult(ServiceFlowResult):
    """Typed result for pair-token verification flow."""

    api_key: Optional[str] = None
    server_id: Optional[str] = None
    client_id: Optional[str] = None
    firebase_registered: Optional[bool] = None
    firebase_error: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("Pairing failed")

        payload: dict[str, Any] = {"success": True}
        if self.api_key:
            payload["api_key"] = self.api_key
        if self.server_id:
            payload["server_id"] = self.server_id
        if self.client_id:
            payload["client_id"] = self.client_id
        if self.firebase_registered is not None:
            payload["firebase_registered"] = self.firebase_registered
        if self.firebase_error:
            payload["firebase_error"] = self.firebase_error
        return payload


@dataclass(frozen=True)
class RemoteAccessActionResult(ServiceFlowResult):
    """Typed result for logout/disconnect remote-access flows."""

    url: Optional[str] = None
    message: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("Remote access action failed")

        payload: dict[str, Any] = {"success": True}
        if self.url:
            payload["url"] = self.url
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class RemoteMdnsStatus:
    """Typed mDNS status for remote-network state responses."""

    available: bool
    enabled: bool
    registered: bool
    server_name: str

    def as_response_fields(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "enabled": self.enabled,
            "registered": self.registered,
            "server_name": self.server_name,
        }


@dataclass(frozen=True)
class RemoteTunnelStatus:
    """Typed tunnel status for remote-network state responses."""

    available: bool
    enabled: bool
    running: bool
    url: Optional[str]
    installed: Optional[bool] = None

    def as_response_fields(self) -> dict[str, Any]:
        payload = {
            "available": self.available,
            "enabled": self.enabled,
            "running": self.running,
            "url": self.url,
        }
        if self.installed is not None:
            payload["installed"] = self.installed
        return payload


@dataclass(frozen=True)
class RemoteFirebaseStatus:
    """Typed Firebase auth status for remote-network state responses."""

    available: bool
    enabled: bool
    authenticated: bool
    user_id: Optional[str]
    device_id: Optional[str]

    def as_response_fields(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "enabled": self.enabled,
            "authenticated": self.authenticated,
            "user_id": self.user_id,
            "device_id": self.device_id,
        }


@dataclass(frozen=True)
class RemoteNetworkStatus:
    """Typed aggregate network/remote-access status payload."""

    mdns: RemoteMdnsStatus
    tunnel: RemoteTunnelStatus
    firebase: RemoteFirebaseStatus

    def as_response_fields(self) -> dict[str, Any]:
        return {
            "mdns": self.mdns.as_response_fields(),
            "tunnel": self.tunnel.as_response_fields(),
            "firebase": self.firebase.as_response_fields(),
        }


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

    raw_auth_mode = body.get("auth_mode", "refresh_token")
    auth_mode = raw_auth_mode if isinstance(raw_auth_mode, str) and raw_auth_mode.strip() else "refresh_token"

    register_device = bool(body.get("register_device", True))

    return (
        RemoteAccessLoginPayload(
            id_token=id_token,
            refresh_token=refresh_token,
            auth_mode=auth_mode,
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
            print("Auto-starting tunnel for remote access...")
            return await tunnel_service.start()
        return tunnel_service.tunnel_url

    if autostart_tunnel:
        tunnel_service = create_tunnel_service(local_port=local_port)
        if tunnel_service:
            print("Creating and starting tunnel for remote access...")
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


async def register_pairing_remote_access(
    *,
    firebase_id_token: str,
    firebase_refresh_token: Optional[str],
    auth_mode: str,
    local_port: int,
) -> PairingRemoteAccessResult:
    """Authenticate Firebase token and register device during pairing."""
    if not firebase_id_token or not FIREBASE_AVAILABLE:
        return PairingRemoteAccessResult()

    firebase_auth = get_firebase_auth()
    if not firebase_auth:
        return PairingRemoteAccessResult()

    auth_success = await firebase_auth.authenticate_with_token(
        firebase_id_token,
        refresh_token=firebase_refresh_token,
        auth_mode=auth_mode,
    )
    if not auth_success:
        return PairingRemoteAccessResult(
            firebase_registered=False,
            firebase_error="Token verification failed",
        )

    registered = await register_device_for_remote_access(
        firebase_auth,
        local_port=local_port,
        autostart_tunnel=True,
    )
    if registered:
        print(f"Server registered to Firebase for user: {firebase_auth.get_status().get('user_id')}")

    return PairingRemoteAccessResult(firebase_registered=registered)


async def login_for_remote_access(
    payload: RemoteAccessLoginPayload,
    *,
    firebase_enabled: bool,
    local_port: int,
    firebase_available: bool = FIREBASE_AVAILABLE,
    firebase_auth: Any | None = None,
) -> RemoteAccessLoginResult:
    """Authenticate Firebase login payload and optionally register the device."""
    if not firebase_available:
        return _firebase_unavailable_login_result()

    if not firebase_enabled:
        return RemoteAccessLoginResult(
            success=False,
            status_code=400,
            error="Firebase is not enabled in config",
        )

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
        auth_mode=payload.auth_mode,
    )
    if not success:
        return RemoteAccessLoginResult(
            success=False,
            status_code=401,
            error="Token verification failed",
        )

    status = resolved_firebase_auth.get_status()
    if payload.register_device:
        await register_device_for_remote_access(
            resolved_firebase_auth,
            local_port=local_port,
            autostart_tunnel=False,
        )

    raw_user_id = status.get("user_id")
    user_id = raw_user_id if isinstance(raw_user_id, str) else None
    raw_device_id = status.get("device_id")
    device_id = raw_device_id if isinstance(raw_device_id, str) else None
    raw_device_name = status.get("device_name")
    device_name = raw_device_name if isinstance(raw_device_name, str) else None
    raw_auth_mode = status.get("auth_mode")
    auth_mode = raw_auth_mode if isinstance(raw_auth_mode, str) and raw_auth_mode else payload.auth_mode

    return RemoteAccessLoginResult(
        success=True,
        status_code=200,
        user_id=user_id,
        device_id=device_id,
        device_name=device_name,
        auth_mode=auth_mode,
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
        firebase_enabled=config.firebase_enabled,
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
    except Exception:
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
    auth_mode: str,
    local_port: int,
) -> PairVerifyFlowResult:
    """Verify pair token and optionally register Firebase remote access."""
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
        remote_result = await register_pairing_remote_access(
            firebase_id_token=firebase_id_token,
            firebase_refresh_token=firebase_refresh_token,
            auth_mode=auth_mode,
            local_port=local_port,
        )
        firebase_registered = remote_result.firebase_registered
        firebase_error = remote_result.firebase_error

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
    auth_mode: str,
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
        auth_mode=auth_mode,
        local_port=config.api_port,
    )


async def verify_pair_token_for_current_server(
    *,
    pair_token: str,
    client_id: Optional[str],
    device_name: Optional[str],
    firebase_id_token: Optional[str],
    firebase_refresh_token: Optional[str],
    auth_mode: str,
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
        auth_mode=auth_mode,
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
    firebase_authenticated, firebase_user_id, firebase_device_id = _extract_firebase_status()

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
            enabled=config.firebase_enabled,
            authenticated=firebase_authenticated,
            user_id=firebase_user_id,
            device_id=firebase_device_id,
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
    firebase_device_id: Optional[str] = None

    if FIREBASE_AVAILABLE:
        firebase_auth = get_firebase_auth()
        if firebase_auth:
            auth_status = firebase_auth.get_status()
            firebase_authenticated = bool(auth_status.get("authenticated", False))
            raw_user_id = auth_status.get("user_id")
            firebase_user_id = raw_user_id if isinstance(raw_user_id, str) else None
            raw_device_id = auth_status.get("device_id")
            firebase_device_id = raw_device_id if isinstance(raw_device_id, str) else None

    return firebase_authenticated, firebase_user_id, firebase_device_id


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
                local_url = f"http://localhost:{local_port}"
                registered = await firebase_auth.register_device(url, local_url)
                if registered:
                    print(f"Tunnel URL registered to Firebase: {url}")

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
    from system_settings_service import get_allow_ip_login

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
