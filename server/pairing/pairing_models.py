"""Data models for pairing operations."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


def _now_ts() -> float:
    """Return current Unix timestamp."""
    return time.time()


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

    def is_connected(self, threshold_seconds: int = 0) -> bool:
        """Check if client is considered connected (last_used within threshold).

        Args:
            threshold_seconds: Custom threshold. If 0, uses heartbeat interval + buffer.
        """
        if self.last_used is None:
            return False

        if threshold_seconds == 0:
            # Use heartbeat interval + 5 min buffer for connectivity check
            from remote.heartbeat_settings import get_heartbeat_interval
            threshold_seconds = (get_heartbeat_interval() + 5) * 60

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
