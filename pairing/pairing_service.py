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

from core.config import get_config
from core.runtime_paths import runtime_dir
from system.optional_services import get_active_tunnel_url, get_server_id as get_firebase_server_id
from .pairing_models import (
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
from core.rate_limiter import (
    RATE_LIMIT_BLOCK_SECONDS,
    RATE_LIMIT_MAX_ATTEMPTS,
    RATE_LIMIT_WINDOW_SECONDS,
    RateLimitEntry,
    RateLimiter,
)

logger = logging.getLogger(__name__)

# Pairing token validity in seconds (5 minutes)
PAIRING_TOKEN_TTL_SECONDS = 300
API_KEY_HASH_FIELD = "api_key_sha256"
LEGACY_API_KEY_FIELD = "api_key"


def _now_ts() -> float:
    return time.time()


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


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
            config_dir = runtime_dir("pairing", Path.home() / ".code-bridge")
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
                data = json.loads(api_keys_file.read_text())
                self._api_keys = data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                self._api_keys = {}

    def _save_api_keys(self, *, allow_empty: bool = False) -> None:
        """Persist API keys to disk.

        Refuses to write an empty store over a file that still has clients
        unless the caller says that is the intent. Every device loses its
        pairing at once when that happens, and the phones cannot tell a wiped
        server from a rejected key — they just stop working and offer a retry
        that can never succeed. Recovering means re-pairing each device by QR.

        The in-memory store is emptied on any read failure in
        :meth:`_load_api_keys`, so one unreadable moment followed by any
        ordinary save was enough to destroy every pairing.
        """
        api_keys_file = self.config_dir / "api_keys.json"

        if not self._api_keys and not allow_empty:
            existing = self._read_api_keys_file(api_keys_file)
            if existing:
                backup = api_keys_file.with_name(
                    f"api_keys.{int(_now_ts())}.recovered.json"
                )
                try:
                    backup.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                    backup.chmod(0o600)
                except OSError:
                    backup = None
                logger.error(
                    "Refusing to overwrite %d paired client(s) with an empty key store. "
                    "Backup: %s",
                    len(existing),
                    backup or "not written",
                )
                # Recover rather than persist the emptiness: the file on disk is
                # the truth here, not whatever cleared memory.
                self._api_keys = existing
                return

        tmp_file = api_keys_file.with_name(f"{api_keys_file.name}.tmp")
        tmp_file.write_text(json.dumps(self._api_keys, indent=2), encoding="utf-8")
        os.replace(tmp_file, api_keys_file)
        try:
            api_keys_file.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _read_api_keys_file(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _store_api_key_hash(self, client_data: dict[str, Any], api_key: str) -> None:
        """Store only a one-way hash of a client API key."""
        client_data[API_KEY_HASH_FIELD] = _hash_api_key(api_key)
        client_data.pop(LEGACY_API_KEY_FIELD, None)

    def _api_key_matches(self, key_data: dict[str, Any], api_key: str) -> bool:
        """Check an API key and opportunistically migrate legacy plaintext rows."""
        stored_hash = key_data.get(API_KEY_HASH_FIELD)
        if isinstance(stored_hash, str):
            return secrets.compare_digest(stored_hash, _hash_api_key(api_key))

        legacy_key = key_data.get(LEGACY_API_KEY_FIELD)
        if isinstance(legacy_key, str) and secrets.compare_digest(legacy_key, api_key):
            self._store_api_key_hash(key_data, api_key)
            return True
        return False

    @property
    def server_id(self) -> str:
        """Get persistent server identifier.

        Uses the Firebase server_id to ensure consistency between pairing
        and Firebase registration. Falls back to a local ID if Firebase
        is not available or not initialized.
        """
        # Use safe server_id accessor that doesn't trigger side effects
        server_id = get_firebase_server_id()
        if server_id:
            return server_id

        # Fallback: generate local server ID (should rarely happen)
        return self._get_or_generate_local_server_id()

    def _get_or_generate_local_server_id(self) -> str:
        """Generate a local server ID as fallback.

        This is used when Firebase is not available. Uses the same
        algorithm as Firebase service to ensure consistency.
        """
        import hashlib
        import platform
        import subprocess

        # Try to get stable machine identifier
        system = platform.system()
        machine_id = None

        try:
            if system == "Darwin":  # macOS
                result = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "IOPlatformUUID" in line:
                            parts = line.split('"')
                            if len(parts) >= 4:
                                machine_id = parts[3]
                                break
            elif system == "Linux":
                for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
                    from pathlib import Path
                    p = Path(path)
                    if p.exists():
                        machine_id = p.read_text().strip()
                        break
        except (subprocess.SubprocessError, OSError):
            pass

        if not machine_id:
            machine_id = f"{platform.node()}-{platform.machine()}-{system}"

        hash_hex = hashlib.sha256(machine_id.encode()).hexdigest()[:32]
        return f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"

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
        validate_result = self.validate_pair_token(pair_token)
        if not validate_result.success:
            return validate_result

        token_data = self._pending_tokens[pair_token]

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
            "device_name": device_name or "Unknown Device",
            "paired_at": _now_ts(),
            "last_used": _now_ts(),
        }
        self._store_api_key_hash(client_data, api_key)

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

    def validate_pair_token(self, pair_token: str) -> PairingVerifyTokenResult:
        """Validate a pairing token without consuming it or issuing an API key."""
        self._cleanup_expired_tokens()

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

        return PairingVerifyTokenResult(
            success=True,
            status_code=200,
            server_id=self.server_id,
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
            if self._api_key_matches(key_data, api_key):
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
            if self._api_key_matches(key_data, api_key):
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

    def find_client_id_by_api_key(self, api_key: str) -> Optional[str]:
        """Resolve which paired client authenticated with `api_key`.

        Used by the push-token route so a device can only ever register a
        token for the pairing it actually authenticated as — never for a
        client_id it merely claims in the request body.
        """
        for client_id, key_data in self._api_keys.items():
            if self._api_key_matches(key_data, api_key):
                return client_id
        return None

    def register_push_token(
        self,
        client_id: str,
        token: str,
        *,
        platform: str = "android",
    ) -> bool:
        """Attach an FCM token to a paired client.

        Tokens rotate — reinstall, cleared app data, a routine Play Services
        refresh — so the same physical device can show up under a new token
        at any time. Registering is therefore idempotent (the same token
        twice keeps one entry) and self-healing (a token that moved to a
        different client_id is removed from its old owner first, so a
        re-pair on a new account doesn't fan a push out to both).
        """
        if client_id not in self._api_keys:
            return False
        clean_token = str(token or "").strip()
        if not clean_token:
            return False

        for other_id, data in self._api_keys.items():
            if other_id == client_id:
                continue
            tokens = data.get("push_tokens")
            if isinstance(tokens, list) and clean_token in tokens:
                data["push_tokens"] = [t for t in tokens if t != clean_token]

        client_data = self._api_keys[client_id]
        tokens = client_data.get("push_tokens")
        tokens = list(tokens) if isinstance(tokens, list) else []
        if clean_token not in tokens:
            tokens.append(clean_token)
        client_data["push_tokens"] = tokens
        client_data["push_platform"] = platform
        self._save_api_keys()
        return True

    def remove_push_token(self, token: str) -> None:
        """Drop a token FCM reported as unregistered.

        Keeping it around would cost a doomed send attempt on every future
        notify for no benefit — an unregistered token does not come back.
        """
        clean_token = str(token or "").strip()
        if not clean_token:
            return
        changed = False
        for data in self._api_keys.values():
            tokens = data.get("push_tokens")
            if isinstance(tokens, list) and clean_token in tokens:
                data["push_tokens"] = [t for t in tokens if t != clean_token]
                changed = True
        if changed:
            self._save_api_keys()

    def all_push_tokens(self) -> list[str]:
        """Every push token across every paired client.

        The inbox is not per-client, so a notify step's push fans out to
        every device that has ever registered a token, the same way the
        stored notification is visible to every paired client that pulls it.
        """
        tokens: list[str] = []
        for data in self._api_keys.values():
            raw = data.get("push_tokens")
            if isinstance(raw, list):
                tokens.extend(t for t in raw if isinstance(t, str) and t)
        return tokens

    def revoke_client(self, client_id: str) -> PairingRevokeResult:
        """Revoke API key for a client."""
        if client_id in self._api_keys:
            del self._api_keys[client_id]
            # Revoking the last client is a real, intended empty store.
            self._save_api_keys(allow_empty=True)
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


def register_push_token_for_current_server(
    api_key: str,
    token: str,
    *,
    platform: str = "android",
    pairing_service: Optional[PairingService] = None,
) -> bool:
    """Register a push token for whichever paired client owns `api_key`.

    Returns False (rather than raising) when the key does not belong to any
    tracked client — callers turn that into a 404, not a crash.
    """
    resolved_pairing_service = pairing_service or get_pairing_service()
    client_id = resolved_pairing_service.find_client_id_by_api_key(api_key)
    if client_id is None:
        return False
    return resolved_pairing_service.register_push_token(client_id, token, platform=platform)


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
    ownership_conflict: bool = False
    current_owner_email: Optional[str] = None

    def as_response_fields(self) -> dict[str, Any]:
        # Handle ownership conflict (not an error, just needs confirmation)
        if self.ownership_conflict:
            payload: dict[str, Any] = {"ownership_conflict": True}
            if self.current_owner_email:
                payload["current_owner_email"] = self.current_owner_email
            return payload

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


async def _register_sso_firestore_device(
    *,
    firebase_auth: Any,
    pairing_service: PairingService,
    user_id: str,
    id_token: str,
) -> tuple[bool, Optional[str]]:
    """Register the server document for the SSO requesting Firebase user."""
    from firebase import device_registration

    project_id = firebase_auth.project_id
    if not project_id:
        return False, "Firebase project ID unavailable"

    try:
        server_id = firebase_auth.server_id
    except Exception as exc:
        logger.warning("Firebase server ID unavailable during SSO: %s", exc)
        return False, "Firebase server ID unavailable"

    config = get_config()
    local_url = f"http://{pairing_service.get_local_ip()}:{config.api_port}"

    try:
        registered = await device_registration.register_device(
            project_id=project_id,
            user_id=user_id,
            server_id=server_id,
            id_token=id_token,
            tunnel_url=get_active_tunnel_url(),
            local_url=local_url,
        )
    except Exception as exc:
        logger.warning("Firestore registration failed during SSO: %s", exc)
        return False, "Firebase registration failed"

    if not registered:
        return False, "Firebase registration failed"
    return True, None


async def verify_sso_pairing_for_current_server(
    *,
    firebase_id_token: str,
    firebase_refresh_token: Optional[str] = None,
    client_id: Optional[str] = None,
    device_name: Optional[str] = None,
    force_replace: bool = False,
    pairing_service: Optional[PairingService] = None,
) -> SSOPairingResult:
    """Verify Firebase SSO and issue API key.

    Multi-account support:
    - First account to pair via QR becomes the primary owner (server_info.json)
    - Additional accounts can pair via SSO and receive API keys
    - All paired accounts are tracked in paired_accounts.json
    - When server URL changes, all accounts receive updates

    Args:
        firebase_id_token: Firebase ID token from the app
        firebase_refresh_token: Firebase refresh token (optional)
        client_id: Optional client identifier
        device_name: Optional device display name
        force_replace: If True, replace primary owner (not just add secondary account)
        pairing_service: Optional pairing service override

    Returns:
        SSOPairingResult with api_key if successful
    """
    from system.optional_services import get_firebase_auth
    from firebase.paired_accounts import get_paired_accounts_manager

    resolved_pairing_service = pairing_service or get_pairing_service()
    firebase_auth = get_firebase_auth()
    paired_accounts = get_paired_accounts_manager()

    if firebase_auth is None:
        return SSOPairingResult(
            success=False,
            status_code=503,
            error="Firebase not configured on server",
        )

    # Ensure Firebase is initialized before verifying token
    if not firebase_auth.is_initialized:
        try:
            await firebase_auth.initialize()
        except Exception as exc:
            logger.warning("Firebase initialization failed during SSO: %s", exc)
            return SSOPairingResult(
                success=False,
                status_code=503,
                error="Firebase initialization failed",
            )

    # Verify the ID token from app
    try:
        user_info = await firebase_auth.verify_id_token(firebase_id_token)
    except Exception as exc:
        logger.warning("Firebase token verification failed during SSO: %s", exc)
        user_info = None
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

    # Check if server has a primary owner
    server_owner_user_id = firebase_auth.user_id

    if not server_owner_user_id:
        # No primary owner - server must be paired via QR first
        return SSOPairingResult(
            success=False,
            status_code=403,
            error="Server not registered to any user. Please pair via QR code first.",
        )

    is_primary_owner = (requesting_user_id == server_owner_user_id)

    if is_primary_owner:
        # Primary owner - update their auth tokens
        try:
            auth_success = await firebase_auth.authenticate_with_token(
                id_token=firebase_id_token,
                refresh_token=firebase_refresh_token,
            )
        except Exception as exc:
            logger.warning("Firebase authentication failed during SSO: %s", exc)
            auth_success = False
        if not auth_success:
            return SSOPairingResult(
                success=False,
                status_code=401,
                error="Token verification failed",
            )
        logger.info("SSO: Primary owner %s re-authenticated", requesting_email)
    elif force_replace:
        # Taking over as primary owner
        logger.info(
            "SSO ownership takeover: %s taking over from %s",
            requesting_email or requesting_user_id,
            firebase_auth.email or server_owner_user_id,
        )
        try:
            auth_success = await firebase_auth.authenticate_with_token(
                id_token=firebase_id_token,
                refresh_token=firebase_refresh_token,
            )
        except Exception as exc:
            logger.warning("Firebase authentication failed during SSO: %s", exc)
            auth_success = False
        if not auth_success:
            return SSOPairingResult(
                success=False,
                status_code=401,
                error="Token verification failed",
            )
    else:
        # Secondary account - just verify token is valid (already done above)
        logger.info(
            "SSO: Adding secondary account %s (primary: %s)",
            requesting_email or requesting_user_id,
            firebase_auth.email or server_owner_user_id,
        )

    registered, registration_error = await _register_sso_firestore_device(
        firebase_auth=firebase_auth,
        pairing_service=resolved_pairing_service,
        user_id=requesting_user_id,
        id_token=firebase_id_token,
    )
    if not registered:
        return SSOPairingResult(
            success=False,
            status_code=502,
            error=registration_error or "Firebase registration failed",
        )

    # Add/update account in paired accounts manager
    paired_accounts.add_or_update_account(
        user_id=requesting_user_id,
        email=requesting_email,
        id_token=firebase_id_token,
        refresh_token=firebase_refresh_token,
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
        "device_name": device_name or "Unknown Device",
        "paired_at": _now_ts(),
        "last_used": _now_ts(),
        "firebase_user": {
            "user_id": requesting_user_id,
            "email": requesting_email,
        },
        "paired_via": "sso",
        "is_primary_owner": is_primary_owner,
    }
    resolved_pairing_service._store_api_key_hash(client_data, api_key)

    resolved_pairing_service._api_keys[resolved_client_id] = client_data
    resolved_pairing_service._save_api_keys()

    logger.info(
        "SSO: Issued API key for %s %s (client: %s)",
        "primary owner" if is_primary_owner else "secondary account",
        requesting_email,
        resolved_client_id,
    )

    return SSOPairingResult(
        success=True,
        status_code=200,
        api_key=api_key,
        server_id=resolved_pairing_service.server_id,
        client_id=resolved_client_id,
    )
