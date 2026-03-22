"""Data models for remote access operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
    server_id: Optional[str] = None
    server_name: Optional[str] = None
    auth_mode: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        if not self.success:
            return self.error_response("Remote access login failed")
        return {
            "success": True,
            "user_id": self.user_id,
            "server_id": self.server_id,
            "server_name": self.server_name,
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
    server_id: Optional[str]

    def as_response_fields(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "enabled": self.enabled,
            "authenticated": self.authenticated,
            "user_id": self.user_id,
            "server_id": self.server_id,
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
