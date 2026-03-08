"""QR Pairing Service for Code Bridge.

Handles QR code generation, pairing token verification, and API key management.
"""

import base64
import hashlib
import json
import os
import secrets
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import get_config
from firebase_auth import get_firebase_auth
from optional_services import get_active_tunnel_url

# Pairing token validity in seconds (5 minutes)
PAIRING_TOKEN_TTL_SECONDS = 300

# Rate limiting settings
RATE_LIMIT_MAX_ATTEMPTS = 5  # Max failed attempts before lockout
RATE_LIMIT_WINDOW_SECONDS = 60  # Time window for tracking attempts
RATE_LIMIT_LOCKOUT_SECONDS = 300  # Lockout duration (5 minutes)


def _now_ts() -> float:
    return time.time()


@dataclass
class RateLimitEntry:
    """Track rate limiting state for one IP."""

    attempts: int = 0
    first_attempt_at: float = 0.0
    locked_until: float = 0.0

    def is_locked(self, now: float) -> bool:
        """Check if this IP is currently locked out."""
        return self.locked_until > now

    def remaining_lockout_seconds(self, now: float) -> int:
        """Get remaining lockout time in seconds."""
        if not self.is_locked(now):
            return 0
        return int(self.locked_until - now)


class RateLimiter:
    """IP-based rate limiter for pairing code verification.

    Tracks failed attempts per IP and enforces lockouts after too many failures.
    """

    def __init__(
        self,
        max_attempts: int = RATE_LIMIT_MAX_ATTEMPTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
        lockout_seconds: int = RATE_LIMIT_LOCKOUT_SECONDS,
    ):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._entries: dict[str, RateLimitEntry] = {}

    def _cleanup_expired(self, now: float) -> None:
        """Remove stale entries that are no longer locked and outside window."""
        cutoff = now - self.window_seconds - self.lockout_seconds
        expired = [
            ip for ip, entry in self._entries.items()
            if entry.first_attempt_at < cutoff and not entry.is_locked(now)
        ]
        for ip in expired:
            del self._entries[ip]

    def check_rate_limit(self, client_ip: str) -> tuple[bool, int]:
        """Check if client IP is rate limited.

        Args:
            client_ip: Client IP address

        Returns:
            Tuple of (is_allowed, remaining_lockout_seconds)
            - is_allowed: True if request should be allowed
            - remaining_lockout_seconds: Seconds until lockout expires (0 if not locked)
        """
        now = _now_ts()
        self._cleanup_expired(now)

        entry = self._entries.get(client_ip)
        if entry is None:
            return (True, 0)

        if entry.is_locked(now):
            return (False, entry.remaining_lockout_seconds(now))

        # Reset if window expired
        if now - entry.first_attempt_at > self.window_seconds:
            entry.attempts = 0
            entry.first_attempt_at = 0.0

        return (True, 0)

    def record_attempt(self, client_ip: str, success: bool) -> None:
        """Record an attempt and update rate limit state.

        Args:
            client_ip: Client IP address
            success: Whether the attempt was successful
        """
        now = _now_ts()

        if success:
            # Clear state on successful attempt
            if client_ip in self._entries:
                del self._entries[client_ip]
            return

        # Record failed attempt
        entry = self._entries.get(client_ip)
        if entry is None:
            entry = RateLimitEntry()
            self._entries[client_ip] = entry

        # Reset window if expired
        if entry.first_attempt_at == 0 or now - entry.first_attempt_at > self.window_seconds:
            entry.attempts = 0
            entry.first_attempt_at = now

        entry.attempts += 1

        # Trigger lockout if max attempts exceeded
        if entry.attempts >= self.max_attempts:
            entry.locked_until = now + self.lockout_seconds
            print(f"[rate_limit] IP {client_ip} locked for {self.lockout_seconds}s after {entry.attempts} failed attempts")

    def get_status(self, client_ip: str) -> dict[str, Any]:
        """Get current rate limit status for an IP."""
        now = _now_ts()
        entry = self._entries.get(client_ip)

        if entry is None:
            return {
                "attempts": 0,
                "max_attempts": self.max_attempts,
                "is_locked": False,
                "remaining_seconds": 0,
            }

        return {
            "attempts": entry.attempts,
            "max_attempts": self.max_attempts,
            "is_locked": entry.is_locked(now),
            "remaining_seconds": entry.remaining_lockout_seconds(now),
        }


@dataclass
class PairingData:
    """QR code pairing payload data."""

    v: int  # Protocol version
    type: str  # Always "codebridge-pair"
    server_id: str  # Server UUID
    name: str  # Server display name
    local_url: str  # Local network URL
    tunnel_url: Optional[str]  # Cloudflare tunnel URL (if available)
    pair_token: str  # One-time pairing token
    expires: int  # Unix timestamp when token expires
    pairing_code: Optional[str] = None  # 6-digit numeric code for manual entry

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "v": self.v,
            "type": self.type,
            "server_id": self.server_id,
            "name": self.name,
            "local_url": self.local_url,
            "pair_token": self.pair_token,
            "expires": self.expires,
        }
        if self.tunnel_url:
            data["tunnel_url"] = self.tunnel_url
        return data

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def to_base64url(self) -> str:
        """Encode as base64url for QR code."""
        json_bytes = self.to_json().encode("utf-8")
        return base64.urlsafe_b64encode(json_bytes).decode("ascii")

    def to_qr_url(self) -> str:
        """Generate codebridge:// URL for QR code."""
        return f"codebridge://pair/{self.to_base64url()}"

    def expires_in_seconds(self, now_ts: Optional[int] = None) -> int:
        """Return remaining token lifetime in seconds."""
        current_ts = int(_now_ts()) if now_ts is None else int(now_ts)
        return self.expires - current_ts

    def to_qr_response(self, now_ts: Optional[int] = None) -> dict[str, Any]:
        """Build route response payload for QR endpoint."""
        return {
            "qr_url": self.to_qr_url(),
            "payload": self.to_dict(),
            "local_url": self.local_url,
            "tunnel_url": self.tunnel_url,
            "expires_in_seconds": self.expires_in_seconds(now_ts),
            "pairing_code": self.pairing_code,
        }


@dataclass(frozen=True)
class PairTokenStatus:
    """Typed status for a pairing token lookup."""

    exists: bool
    used: bool
    expired: bool

    def as_response_fields(self) -> dict[str, bool]:
        return {
            "exists": self.exists,
            "used": self.used,
            "expired": self.expired,
        }


@dataclass(frozen=True)
class PairingOperationResult:
    """Common base result for pairing operations."""

    success: bool
    status_code: int
    error: Optional[str] = None

    def error_response(self, fallback_message: str) -> dict[str, str]:
        return {"error": self.error or fallback_message}


@dataclass(frozen=True)
class CurrentPairingDataResult(PairingOperationResult):
    """Typed result for current server pairing-data generation."""

    pairing_data: Optional[PairingData] = None


@dataclass(frozen=True)
class PairingVerifyTokenResult(PairingOperationResult):
    """Typed result for pair-token verification."""

    api_key: Optional[str] = None
    server_id: Optional[str] = None
    client_id: Optional[str] = None

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
        return payload


@dataclass(frozen=True)
class PairingRevokeResult(PairingOperationResult):
    """Typed result for paired-client revocation."""

    message: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("Failed to revoke paired client")

        payload: dict[str, Any] = {"success": True}
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class PairingQrResult(PairingOperationResult):
    """Typed result for current QR payload generation."""

    qr_url: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    local_url: Optional[str] = None
    tunnel_url: Optional[str] = None
    expires_in_seconds: Optional[int] = None
    pairing_code: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("Failed to build pairing QR data")

        return {
            "qr_url": self.qr_url,
            "payload": self.payload,
            "local_url": self.local_url,
            "tunnel_url": self.tunnel_url,
            "expires_in_seconds": self.expires_in_seconds,
            "pairing_code": self.pairing_code,
        }


@dataclass(frozen=True)
class PairingPageContextResult(PairingOperationResult):
    """Typed result for pairing page render context."""

    qr_url: Optional[str] = None
    local_url: Optional[str] = None
    pair_token: Optional[str] = None
    expires_in_seconds: Optional[int] = None
    pairing_code: Optional[str] = None

    def to_render_context(self) -> tuple[str, str, str, int, str] | None:
        """Return validated HTML render context on success."""
        if not self.success:
            return None
        if not isinstance(self.qr_url, str):
            return None
        if not isinstance(self.local_url, str):
            return None
        if not isinstance(self.pair_token, str):
            return None
        expires_in_seconds = self.expires_in_seconds if isinstance(self.expires_in_seconds, int) else 0
        pairing_code = self.pairing_code if isinstance(self.pairing_code, str) else ""
        return (self.qr_url, self.local_url, self.pair_token, expires_in_seconds, pairing_code)

    def to_html_error(self) -> tuple[str, int]:
        """Return fallback HTML error payload for invalid/failed page context."""
        if not self.success:
            return (self.error or "Failed to build pairing page", self.status_code)
        return ("Failed to build pairing page", 500)


@dataclass(frozen=True)
class FirebaseUserInfo:
    """Firebase user info associated with a paired client."""

    user_id: Optional[str] = None
    email: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
        }


@dataclass(frozen=True)
class PairingClientStatus:
    """Typed status for one paired client."""

    client_id: str
    device_name: str
    paired_at: Optional[float]
    last_used: Optional[float]
    firebase_user: Optional[FirebaseUserInfo] = None

    def is_connected(self, threshold_seconds: int = 30) -> bool:
        """Check if client is considered connected (last_used within threshold)."""
        if self.last_used is None:
            return False
        return (_now_ts() - self.last_used) < threshold_seconds

    def as_response_fields(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "client_id": self.client_id,
            "device_name": self.device_name,
            "paired_at": self.paired_at,
            "last_used": self.last_used,
            "is_connected": self.is_connected(),
        }
        if self.firebase_user:
            result["firebase_user"] = self.firebase_user.as_response_fields()
        return result


@dataclass(frozen=True)
class PairingStatus:
    """Typed aggregate pairing status."""

    server_id: str
    active_clients: int
    pending_tokens: int
    clients: list[PairingClientStatus]

    def as_response_fields(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "active_clients": self.active_clients,
            "pending_tokens": self.pending_tokens,
            "clients": [client.as_response_fields() for client in self.clients],
        }


@dataclass(frozen=True)
class PairingCodeVerifyResult(PairingOperationResult):
    """Typed result for numeric code verification."""

    pair_token: Optional[str] = None
    server_id: Optional[str] = None
    local_url: Optional[str] = None
    tunnel_url: Optional[str] = None
    expires: Optional[int] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("Invalid code")
        result: dict[str, Any] = {"success": True, "pair_token": self.pair_token}
        if self.server_id:
            result["server_id"] = self.server_id
        if self.local_url:
            result["local_url"] = self.local_url
        if self.tunnel_url:
            result["tunnel_url"] = self.tunnel_url
        if self.expires:
            result["expires"] = self.expires
        return result


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
        except Exception:
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
            local_url=f"http://{self.get_local_ip()}:{get_config().port}",
            tunnel_url=get_active_tunnel_url(),
            expires=token_data["expires_at"],
        )

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

        # Generate API key for this client
        api_key = self._generate_api_key()
        resolved_client_id = client_id or str(uuid.uuid4())

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
    except Exception:
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
    except Exception:
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
    except Exception:
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

    # Generate API key for this client
    resolved_client_id = client_id or str(uuid.uuid4())
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

    print(f"[SSO Pairing] Issued API key for user {requesting_email} (client: {resolved_client_id})")

    return SSOPairingResult(
        success=True,
        status_code=200,
        api_key=api_key,
        server_id=resolved_pairing_service.server_id,
        client_id=resolved_client_id,
    )
