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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Pairing token validity in seconds (5 minutes)
PAIRING_TOKEN_TTL_SECONDS = 300


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

        self._server_id: Optional[str] = None
        self._api_keys: dict[str, dict[str, Any]] = {}  # client_id -> key info
        self._pending_tokens: dict[str, dict[str, Any]] = {}  # token -> metadata

        self._load_server_id()
        self._load_api_keys()

    def _load_server_id(self) -> None:
        """Load or generate persistent server ID."""
        server_id_file = self.config_dir / "server_id"
        if server_id_file.exists():
            self._server_id = server_id_file.read_text().strip()
        else:
            self._server_id = str(uuid.uuid4())
            server_id_file.write_text(self._server_id)

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
        """Get persistent server identifier."""
        if self._server_id is None:
            self._load_server_id()
        return self._server_id or str(uuid.uuid4())

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
        expires = int(time.time()) + PAIRING_TOKEN_TTL_SECONDS

        # Store pending token
        self._pending_tokens[pair_token] = {
            "created_at": time.time(),
            "expires_at": expires,
            "used": False,
        }

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
        )

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired pairing tokens."""
        now = time.time()
        expired = [
            token
            for token, data in self._pending_tokens.items()
            if data["expires_at"] < now
        ]
        for token in expired:
            del self._pending_tokens[token]

    def verify_pair_token(
        self,
        pair_token: str,
        client_id: Optional[str] = None,
        device_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Verify a pairing token and issue an API key.

        Args:
            pair_token: The pairing token from QR code
            client_id: Optional client identifier
            device_name: Optional device display name

        Returns:
            Dict with success status and API key if valid
        """
        self._cleanup_expired_tokens()

        # Check if token exists and is valid
        token_data = self._pending_tokens.get(pair_token)
        if token_data is None:
            return {"success": False, "error": "Invalid or expired token"}

        if token_data["used"]:
            return {"success": False, "error": "Token already used"}

        if token_data["expires_at"] < time.time():
            del self._pending_tokens[pair_token]
            return {"success": False, "error": "Token expired"}

        # Mark token as used
        token_data["used"] = True

        # Generate API key for this client
        api_key = self._generate_api_key()
        resolved_client_id = client_id or str(uuid.uuid4())

        # Store API key
        self._api_keys[resolved_client_id] = {
            "api_key": api_key,
            "device_name": device_name or "Unknown Device",
            "paired_at": time.time(),
            "last_used": time.time(),
        }
        self._save_api_keys()

        return {
            "success": True,
            "api_key": api_key,
            "server_id": self.server_id,
            "client_id": resolved_client_id,
        }

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
                key_data["last_used"] = time.time()
                self._save_api_keys()
                return True
        return False

    def get_pairing_status(self) -> dict[str, Any]:
        """Get current pairing status.

        Returns:
            Dict with pairing information
        """
        self._cleanup_expired_tokens()
        active_clients = len(self._api_keys)
        pending_tokens = len(
            [t for t, d in self._pending_tokens.items() if not d["used"]]
        )

        return {
            "server_id": self.server_id,
            "active_clients": active_clients,
            "pending_tokens": pending_tokens,
            "clients": [
                {
                    "client_id": cid,
                    "device_name": data.get("device_name", "Unknown"),
                    "paired_at": data.get("paired_at"),
                    "last_used": data.get("last_used"),
                }
                for cid, data in self._api_keys.items()
            ],
        }

    def revoke_client(self, client_id: str) -> bool:
        """Revoke API key for a client.

        Args:
            client_id: Client identifier to revoke

        Returns:
            True if revoked, False if not found
        """
        if client_id in self._api_keys:
            del self._api_keys[client_id]
            self._save_api_keys()
            return True
        return False


# Singleton instance
_pairing_service: Optional[PairingService] = None


def get_pairing_service() -> PairingService:
    """Get the singleton pairing service instance."""
    global _pairing_service
    if _pairing_service is None:
        _pairing_service = PairingService()
    return _pairing_service
