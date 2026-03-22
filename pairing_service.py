"""QR Pairing Service for Code Bridge.

Handles QR code generation, pairing token verification, and API key management.
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config import get_config
from firebase_auth import get_firebase_auth
from optional_services import get_active_tunnel_url
from pairing_models import (
    CurrentPairingDataResult,
    FirebaseUserInfo,
    PairingClientStatus,
    PairingCodeVerifyResult,
    PairingData,
    PairingOperationResult,
    PairingPageContextResult,
    PairingQrResult,
    PairingRevokeResult,
    PairingStatus,
    PairingVerifyTokenResult,
    PairTokenStatus,
)
from rate_limiter import (
    RATE_LIMIT_BLOCK_SECONDS,
    RATE_LIMIT_MAX_ATTEMPTS,
    RATE_LIMIT_WINDOW_SECONDS,
    RateLimitEntry,
    RateLimiter,
)

logger = logging.getLogger(__name__)

# Pairing token validity in seconds (5 minutes)
PAIRING_TOKEN_TTL_SECONDS = 300


def _now_ts() -> float:
    return time.time()


# Backward compatibility aliases
RATE_LIMIT_LOCKOUT_SECONDS = RATE_LIMIT_BLOCK_SECONDS


# NOTE: Model classes (PairingData, PairTokenStatus, etc.) are imported from pairing_models.py
# NOTE: RateLimiter and RateLimitEntry are imported from rate_limiter.py


class PairingService:
    """Manages QR-based pairing between app and server."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize pairing service.

        Args:
            config_dir: Directory for storing pairing state and API keys.
                       Defaults to ~/.code-bridge/
        """
        if config_dir is None:
            config_dir = Path.home() / ".code-bridge"
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._api_keys: dict[str, dict[str, Any]] = {}  # client_id -> key info
        self._pending_tokens: dict[str, dict[str, Any]] = {}  # token -> metadata
        self._pending_codes: dict[str, str] = {}  # 6-digit code -> pair_token
        self._rate_limiter = RateLimiter()  # Rate limiter for code verification

        self._load_api_keys()

    def _load_api_keys(self) -> None:
        """Load registered API keys from disk."""
        api_keys_file = self.config_dir / "api_keys.json"
        if api_keys_file.exists():
            try:
                self._api_keys = json.loads(api_keys_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._api_keys = {}

    def _save_api_keys(self) -> None:
        """Persist API keys to disk."""
        api_keys_file = self.config_dir / "api_keys.json"
        api_keys_file.write_text(json.dumps(self._api_keys, indent=2))

    @property
    def server_id(self) -> str:
        """Get persistent server identifier.

        Uses the Firebase server_id to ensure consistency between pairing
        and Firebase registration.
        """
        # Use Firebase server_id for consistency with Firebase registration
        return get_firebase_auth().server_id

    def get_local_ip(self) -> str:
        """Get local network IP address."""
        try:
            # Create a socket to determine local IP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.5)
            # Connect to external address (doesn't actually send data)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except OSError as exc:
            logger.debug("Could not determine local IP: %s, using fallback", exc)
            return "127.0.0.1"

    def generate_pair_token(self) -> str:
        """Generate a secure one-time pairing token."""
        return secrets.token_hex(16)  # 32 character hex string

    def generate_pairing_code(self) -> str:
        """Generate a 6-digit numeric pairing code."""
        return f"{secrets.randbelow(1000000):06d}"

    def create_pairing_data(
        self,
        port: int = 8080,
        server_name: Optional[str] = None,
        tunnel_url: Optional[str] = None,
    ) -> PairingData:
        """Create QR pairing data with a fresh token.

        Args:
            port: Server port number
            server_name: Display name for the server
            tunnel_url: Cloudflare tunnel URL if available

        Returns:
            PairingData object ready for QR code generation
        """
        local_ip = self.get_local_ip()
        pair_token = self.generate_pair_token()
        expires = int(_now_ts()) + PAIRING_TOKEN_TTL_SECONDS

        # Store pending token
        self._pending_tokens[pair_token] = {
            "created_at": _now_ts(),
            "expires_at": expires,
            "used": False,
        }

        # Generate and store 6-digit code for this token
        pairing_code = self.generate_pairing_code()
        self._pending_codes[pairing_code] = pair_token

        # Clean up expired tokens
        self._cleanup_expired_tokens()

        if server_name is None:
            server_name = socket.gethostname() or "PC Server"

        return PairingData(
            v=1,
            type="codebridge-pair",
            server_id=self.server_id,
            name=server_name,
            local_url=f"http://{local_ip}:{port}",
            tunnel_url=tunnel_url,
            pair_token=pair_token,
            expires=expires,
            pairing_code=pairing_code,
        )

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired pairing tokens and their associated codes."""
        now = _now_ts()
        expired = [
            token
            for token, data in self._pending_tokens.items()
            if data["expires_at"] < now
        ]
        for token in expired:
            del self._pending_tokens[token]

        # Clean up codes that point to expired/deleted tokens
        codes_to_remove = [
            code for code, token in self._pending_codes.items()
            if token not in self._pending_tokens
        ]
        for code in codes_to_remove:
            del self._pending_codes[code]

    def verify_pairing_code(
        self,
        code: str,
        client_ip: Optional[str] = None,
    ) -> PairingCodeVerifyResult:
        """Verify a 6-digit pairing code and return the associated pair_token.

        Args:
            code: 6-digit numeric code
            client_ip: Client IP address for rate limiting (optional)

        Returns:
            Typed verification result with pair_token if valid
        """
        # Check rate limit if client_ip provided
        if client_ip:
            is_allowed, remaining_seconds = self._rate_limiter.check_rate_limit(client_ip)
            if not is_allowed:
                return PairingCodeVerifyResult(
                    success=False,
                    status_code=429,
                    error=f"Too many attempts. Try again in {remaining_seconds} seconds.",
                )

        self._cleanup_expired_tokens()

        # Normalize code (strip whitespace)
        code = code.strip()

        pair_token = self._pending_codes.get(code)
        if pair_token is None:
            if client_ip:
                self._rate_limiter.record_attempt(client_ip, success=False)
            return PairingCodeVerifyResult(
                success=False,
                status_code=400,
                error="Invalid code",
            )

        # Check if the token is still valid
        token_data = self._pending_tokens.get(pair_token)
        if token_data is None:
            del self._pending_codes[code]
            if client_ip:
                self._rate_limiter.record_attempt(client_ip, success=False)
            return PairingCodeVerifyResult(
                success=False,
                status_code=400,
                error="Code expired",
            )

        if token_data["used"]:
            if client_ip:
                self._rate_limiter.record_attempt(client_ip, success=False)
            return PairingCodeVerifyResult(
                success=False,
                status_code=400,
                error="Code already used",
            )

        if token_data["expires_at"] < _now_ts():
            if client_ip:
                self._rate_limiter.record_attempt(client_ip, success=False)
            return PairingCodeVerifyResult(
                success=False,
                status_code=400,
                error="Code expired",
            )

        # Success - clear rate limit state
        if client_ip:
            self._rate_limiter.record_attempt(client_ip, success=True)

        return PairingCodeVerifyResult(
            success=True,
            status_code=200,
            pair_token=pair_token,
            server_id=self.server_id,
            local_url=f"http://{self.get_local_ip()}:{get_config().api_port}",
            tunnel_url=get_active_tunnel_url(),
            expires=token_data["expires_at"],
        )

    def _find_client_by_device_name(self, device_name: str) -> Optional[str]:
        """Find existing client_id by device_name.

        Returns client_id if found, None otherwise.
        """
        if not device_name:
            return None

        for cid, data in self._api_keys.items():
            if data.get("device_name") == device_name:
                return cid
        return None

    def verify_pair_token(
        self,
        pair_token: str,
        client_id: Optional[str] = None,
        device_name: Optional[str] = None,
        firebase_user_id: Optional[str] = None,
        firebase_email: Optional[str] = None,
    ) -> PairingVerifyTokenResult:
        """Verify a pairing token and issue an API key.

        Args:
            pair_token: The pairing token from QR code
            client_id: Optional client identifier
            device_name: Optional device display name
            firebase_user_id: Optional Firebase user ID (for integrated display)
            firebase_email: Optional Firebase user email (for integrated display)

        Returns:
            Typed verification result
        """
        self._cleanup_expired_tokens()

        # Check if token exists and is valid
        token_data = self._pending_tokens.get(pair_token)
        if token_data is None:
            return PairingVerifyTokenResult(
                success=False,
                status_code=400,
                error="Invalid or expired token",
            )

        if token_data["used"]:
            return PairingVerifyTokenResult(
                success=False,
                status_code=400,
                error="Token already used",
            )

        if token_data["expires_at"] < _now_ts():
            del self._pending_tokens[pair_token]
            return PairingVerifyTokenResult(
                success=False,
                status_code=400,
                error="Token expired",
            )

        # Mark token as used
        token_data["used"] = True

        # Check for existing client with same device_name (prevents duplicate registrations)
        existing_client_id = self._find_client_by_device_name(device_name or "")
        if existing_client_id and existing_client_id != client_id:
            # Update existing client instead of creating new one
            logger.info("Updating existing client %s (device: %s)", existing_client_id, device_name)
            resolved_client_id = existing_client_id
        else:
            resolved_client_id = client_id or str(uuid.uuid4())

        # Generate API key for this client
        api_key = self._generate_api_key()

        # Store API key with optional Firebase user info
        client_data: dict[str, Any] = {
            "api_key": api_key,
            "device_name": device_name or "Unknown Device",
            "paired_at": _now_ts(),
            "last_used": _now_ts(),
        }

        # Include Firebase user info if provided
        if firebase_user_id or firebase_email:
            client_data["firebase_user"] = {
                "user_id": firebase_user_id,
                "email": firebase_email,
            }

        self._api_keys[resolved_client_id] = client_data
        self._save_api_keys()

        return PairingVerifyTokenResult(
            success=True,
            status_code=200,
            api_key=api_key,
            server_id=self.server_id,
            client_id=resolved_client_id,
        )

    def update_client_firebase_user(
        self,
        client_id: str,
        firebase_user_id: Optional[str] = None,
        firebase_email: Optional[str] = None,
    ) -> bool:
        """Update Firebase user info for an existing client.

        Returns True if client was found and updated.
        """
        if client_id not in self._api_keys:
            return False

        self._api_keys[client_id]["firebase_user"] = {
            "user_id": firebase_user_id,
            "email": firebase_email,
        }
        self._save_api_keys()
        return True

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        # 32 bytes = 256 bits of entropy
        raw_key = secrets.token_bytes(32)
        return base64.urlsafe_b64encode(raw_key).decode("ascii").rstrip("=")

    def validate_api_key(self, api_key: str) -> bool:
        """Check if an API key is valid.

        Args:
            api_key: The API key to validate

        Returns:
            True if valid, False otherwise
        """
        for client_id, key_data in self._api_keys.items():
            if key_data.get("api_key") == api_key:
                # Update last used timestamp
                key_data["last_used"] = _now_ts()
                self._save_api_keys()
                return True
        return False

    def touch_api_key(self, api_key: str) -> bool:
        """Update last_used timestamp for an API key without saving to disk.

        Use this for frequent updates like WebSocket pings to avoid disk I/O.

        Args:
            api_key: The API key to touch

        Returns:
            True if key found and updated, False otherwise
        """
        for client_id, key_data in self._api_keys.items():
            if key_data.get("api_key") == api_key:
                key_data["last_used"] = _now_ts()
                return True
        return False

    def get_pairing_status(self) -> PairingStatus:
        """Get current pairing status.

        Returns:
            Typed pairing status
        """
        self._cleanup_expired_tokens()
        active_clients = len(self._api_keys)
        pending_tokens = len(
            [t for t, d in self._pending_tokens.items() if not d["used"]]
        )

        clients: list[PairingClientStatus] = []
        for cid, data in self._api_keys.items():
            raw_device_name = data.get("device_name")
            device_name = raw_device_name if isinstance(raw_device_name, str) else "Unknown"

            raw_paired_at = data.get("paired_at")
            paired_at = float(raw_paired_at) if isinstance(raw_paired_at, (int, float)) else None

            raw_last_used = data.get("last_used")
            last_used = float(raw_last_used) if isinstance(raw_last_used, (int, float)) else None

            # Extract Firebase user info if present
            firebase_user = None
            raw_firebase_user = data.get("firebase_user")
            if isinstance(raw_firebase_user, dict):
                firebase_user = FirebaseUserInfo(
                    user_id=raw_firebase_user.get("user_id"),
                    email=raw_firebase_user.get("email"),
                )

            clients.append(
                PairingClientStatus(
                    client_id=cid,
                    device_name=device_name,
                    paired_at=paired_at,
                    last_used=last_used,
                    firebase_user=firebase_user,
                )
            )

        return PairingStatus(
            server_id=self.server_id,
            active_clients=active_clients,
            pending_tokens=pending_tokens,
            clients=clients,
        )

    def get_pair_token_status(self, pair_token: str) -> PairTokenStatus:
        """Get status flags for a pairing token."""
        self._cleanup_expired_tokens()
        token_data = self._pending_tokens.get(pair_token)
        if token_data is None:
            return PairTokenStatus(exists=False, used=False, expired=True)

        return PairTokenStatus(
            exists=True,
            used=bool(token_data.get("used", False)),
            expired=token_data.get("expires_at", 0) < _now_ts(),
        )

    def revoke_client(self, client_id: str) -> PairingRevokeResult:
        """Revoke API key for a client."""
        if client_id in self._api_keys:
            del self._api_keys[client_id]
            self._save_api_keys()
            return PairingRevokeResult(
                success=True,
                status_code=200,
                message=f"Client {client_id} revoked",
            )
        return PairingRevokeResult(
            success=False,
            status_code=404,
            error=f"Client {client_id} not found",
        )


def _normalize_qr_response(response: dict[str, Any]) -> PairingQrResult:
    raw_qr_url = response.get("qr_url")
    raw_payload = response.get("payload")
    raw_local_url = response.get("local_url")
    raw_tunnel_url = response.get("tunnel_url")
    raw_expires = response.get("expires_in_seconds")
    raw_pairing_code = response.get("pairing_code")

    return PairingQrResult(
        success=True,
        status_code=200,
        qr_url=raw_qr_url if isinstance(raw_qr_url, str) else None,
        payload=raw_payload if isinstance(raw_payload, dict) else {},
        local_url=raw_local_url if isinstance(raw_local_url, str) else None,
        tunnel_url=raw_tunnel_url if isinstance(raw_tunnel_url, str) or raw_tunnel_url is None else None,
        expires_in_seconds=raw_expires if isinstance(raw_expires, int) else None,
        pairing_code=raw_pairing_code if isinstance(raw_pairing_code, str) else None,
    )


def create_current_pairing_data(
    *,
    port: int,
    server_name: str,
    tunnel_url: Optional[str],
    pairing_service: Optional[PairingService] = None,
) -> PairingData:
    """Create pairing payload for current server state."""
    resolved_pairing_service = pairing_service or get_pairing_service()
    return resolved_pairing_service.create_pairing_data(
        port=port,
        server_name=server_name,
        tunnel_url=tunnel_url,
    )


def build_current_pairing_qr_result(
    *,
    pairing_service: Optional[PairingService] = None,
) -> PairingQrResult:
    """Build typed QR response payload for current server state."""
    pairing_data_result = build_current_pairing_data_result(pairing_service=pairing_service)
    if not pairing_data_result.success or pairing_data_result.pairing_data is None:
        return PairingQrResult(
            success=False,
            status_code=pairing_data_result.status_code,
            error=pairing_data_result.error or "Failed to build pairing QR data",
        )

    pairing_data = pairing_data_result.pairing_data
    try:
        return _normalize_qr_response(pairing_data.to_qr_response())
    except (ValueError, TypeError, AttributeError) as exc:
        logger.exception("Failed to build pairing QR data: %s", exc)
        return PairingQrResult(
            success=False,
            status_code=500,
            error="Failed to build pairing QR data",
        )


def build_current_pairing_page_context_result(
    *,
    pairing_service: Optional[PairingService] = None,
) -> PairingPageContextResult:
    """Build typed render context for the pairing web page."""
    pairing_data_result = build_current_pairing_data_result(pairing_service=pairing_service)
    if not pairing_data_result.success or pairing_data_result.pairing_data is None:
        return PairingPageContextResult(
            success=False,
            status_code=pairing_data_result.status_code,
            error=pairing_data_result.error or "Failed to build pairing page context",
        )

    pairing_data = pairing_data_result.pairing_data
    try:
        return PairingPageContextResult(
            success=True,
            status_code=200,
            qr_url=pairing_data.to_qr_url(),
            local_url=pairing_data.local_url,
            pair_token=pairing_data.pair_token,
            expires_in_seconds=pairing_data.expires_in_seconds(),
            pairing_code=pairing_data.pairing_code,
        )
    except (ValueError, TypeError, AttributeError) as exc:
        logger.exception("Failed to build pairing page context: %s", exc)
        return PairingPageContextResult(
            success=False,
            status_code=500,
            error="Failed to build pairing page context",
        )


def build_current_pairing_data_result(
    *,
    pairing_service: Optional[PairingService] = None,
) -> CurrentPairingDataResult:
    """Build typed current pairing data using config and tunnel context."""
    try:
        config = get_config()
        pairing_data = create_current_pairing_data(
            port=config.api_port,  # App connects to API server
            server_name=config.server_name,
            tunnel_url=get_active_tunnel_url(),
            pairing_service=pairing_service,
        )
        return CurrentPairingDataResult(
            success=True,
            status_code=200,
            pairing_data=pairing_data,
        )
    except Exception as exc:
        logger.exception("Failed to build pairing data: %s", exc)
        return CurrentPairingDataResult(
            success=False,
            status_code=500,
            error="Failed to build pairing data",
        )


def get_pair_token_status_for_current_server(
    pair_token: str,
    *,
    pairing_service: Optional[PairingService] = None,
) -> PairTokenStatus:
    """Get pair-token status using current pairing service context."""
    resolved_pairing_service = pairing_service or get_pairing_service()
    return resolved_pairing_service.get_pair_token_status(pair_token)


def get_pairing_status_for_current_server(
    *,
    pairing_service: Optional[PairingService] = None,
) -> PairingStatus:
    """Get pairing status using current pairing service context."""
    resolved_pairing_service = pairing_service or get_pairing_service()
    return resolved_pairing_service.get_pairing_status()


def revoke_paired_client_for_current_server(
    client_id: str,
    *,
    pairing_service: Optional[PairingService] = None,
) -> PairingRevokeResult:
    """Revoke paired client using current pairing service context."""
    resolved_pairing_service = pairing_service or get_pairing_service()
    return resolved_pairing_service.revoke_client(client_id)


def verify_pairing_code_for_current_server(
    code: str,
    *,
    client_ip: Optional[str] = None,
    pairing_service: Optional[PairingService] = None,
) -> PairingCodeVerifyResult:
    """Verify pairing code using current pairing service context.

    Args:
        code: 6-digit numeric code
        client_ip: Client IP address for rate limiting
        pairing_service: Optional pairing service override

    Returns:
        Typed verification result
    """
    resolved_pairing_service = pairing_service or get_pairing_service()
    return resolved_pairing_service.verify_pairing_code(code, client_ip=client_ip)


# Singleton instance
_pairing_service: Optional[PairingService] = None


def get_pairing_service() -> PairingService:
    """Get the singleton pairing service instance."""
    global _pairing_service
    if _pairing_service is None:
        _pairing_service = PairingService()
    return _pairing_service


@dataclass(frozen=True)
class SSOPairingResult(PairingOperationResult):
    """Typed result for SSO-based pairing."""

    api_key: Optional[str] = None
    server_id: Optional[str] = None
    client_id: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("SSO pairing failed")

        payload: dict[str, Any] = {"success": True}
        if self.api_key:
            payload["api_key"] = self.api_key
        if self.server_id:
            payload["server_id"] = self.server_id
        if self.client_id:
            payload["client_id"] = self.client_id
        return payload


async def verify_sso_pairing_for_current_server(
    *,
    firebase_id_token: str,
    firebase_refresh_token: Optional[str] = None,
    auth_mode: str = "refresh_token",
    client_id: Optional[str] = None,
    device_name: Optional[str] = None,
    pairing_service: Optional[PairingService] = None,
) -> SSOPairingResult:
    """Verify Firebase SSO and issue API key if user owns this server.

    This endpoint is called when app selects a remote server from Firebase.
    Server verifies the ID token and checks if the requesting user matches
    the server's registered owner before issuing an API key.

    Args:
        firebase_id_token: Firebase ID token from the app
        firebase_refresh_token: Firebase refresh token (optional)
        auth_mode: "id_token" (1hr) or "refresh_token" (permanent)
        client_id: Optional client identifier
        device_name: Optional device display name
        pairing_service: Optional pairing service override

    Returns:
        SSOPairingResult with api_key if successful
    """
    from optional_services import get_firebase_auth

    resolved_pairing_service = pairing_service or get_pairing_service()
    firebase_auth = get_firebase_auth()

    if firebase_auth is None:
        return SSOPairingResult(
            success=False,
            status_code=503,
            error="Firebase not configured on server",
        )

    # Verify the ID token from app
    user_info = await firebase_auth.verify_id_token(firebase_id_token)
    if not user_info:
        return SSOPairingResult(
            success=False,
            status_code=401,
            error="Invalid Firebase ID token",
        )

    requesting_user_id = user_info.get("user_id")
    requesting_email = user_info.get("email")

    if not requesting_user_id:
        return SSOPairingResult(
            success=False,
            status_code=401,
            error="Invalid user ID in token",
        )

    # Check if requesting user owns this server
    # Server's owner is stored in device_info.json (user_id field)
    server_owner_user_id = firebase_auth._current_user_id

    if not server_owner_user_id:
        return SSOPairingResult(
            success=False,
            status_code=403,
            error="Server not registered to any user. Please pair via QR code first.",
        )

    if requesting_user_id != server_owner_user_id:
        return SSOPairingResult(
            success=False,
            status_code=403,
            error="You do not own this server",
        )

    # User owns this server - authenticate and issue API key
    await firebase_auth.authenticate_with_token(
        id_token=firebase_id_token,
        refresh_token=firebase_refresh_token,
        auth_mode=auth_mode,
    )

    # Check for existing client with same device_name (prevents duplicate registrations)
    existing_client_id = resolved_pairing_service._find_client_by_device_name(device_name or "")
    if existing_client_id and existing_client_id != client_id:
        # Update existing client instead of creating new one
        logger.info("SSO: Updating existing client %s (device: %s)", existing_client_id, device_name)
        resolved_client_id = existing_client_id
    else:
        resolved_client_id = client_id or str(uuid.uuid4())

    # Generate API key for this client
    api_key = resolved_pairing_service._generate_api_key()

    # Store API key with Firebase user info
    client_data: dict[str, Any] = {
        "api_key": api_key,
        "device_name": device_name or "Unknown Device",
        "paired_at": _now_ts(),
        "last_used": _now_ts(),
        "firebase_user": {
            "user_id": requesting_user_id,
            "email": requesting_email,
        },
        "paired_via": "sso",  # Mark as SSO-paired
    }

    resolved_pairing_service._api_keys[resolved_client_id] = client_data
    resolved_pairing_service._save_api_keys()

    logger.info("SSO: Issued API key for user %s (client: %s)", requesting_email, resolved_client_id)

    return SSOPairingResult(
        success=True,
        status_code=200,
        api_key=api_key,
        server_id=resolved_pairing_service.server_id,
        client_id=resolved_client_id,
    )
