"""Firebase Authentication Service with explicit state management.

This module provides the main FirebaseAuthService class with:
- Explicit initialization requirement
- Read-only properties that never cause side effects
- Immutable state through PersistedState
- Clear state machine transitions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.runtime_paths import SERVER_DIR, runtime_path

from .state import AuthState, LegacyDeviceInfo, PersistedState
from .token_manager import TokenManager, TokenInfo, TOKEN_REFRESH_THRESHOLD_SECONDS
from . import device_registration

logger = logging.getLogger(__name__)

# Default paths
FIREBASE_CONFIG_PATH = runtime_path("firebase_config.json", SERVER_DIR / "firebase_config.json")
SERVER_INFO_PATH = runtime_path("server_info.json", SERVER_DIR / "server_info.json")
LEGACY_DEVICE_INFO_PATH = runtime_path("device_info.json", SERVER_DIR / "device_info.json")


class FirebaseAuthServiceError(Exception):
    """Base exception for FirebaseAuthService errors."""
    pass


class NotInitializedError(FirebaseAuthServiceError):
    """Raised when accessing service before initialize() is called."""
    pass


class FirebaseAuthService:
    """Manages Firebase token verification and Firestore operations.

    This service verifies ID tokens sent from the app and uses the verified
    user information for device registration and Firestore access.

    IMPORTANT: You MUST call initialize() before using any other methods.
    Properties are read-only and never cause side effects.

    State Machine:
    - UNINITIALIZED: Service created, not started
    - LOADING: Loading config/state from disk
    - LOADED: State loaded, but not authenticated
    - AUTHENTICATED: Token verified, user authenticated
    - ERROR: Unrecoverable error state
    """

    def __init__(
        self,
        config_path: Path = FIREBASE_CONFIG_PATH,
        state_path: Path = SERVER_INFO_PATH,
        legacy_path: Path = LEGACY_DEVICE_INFO_PATH,
    ):
        """Initialize Firebase auth service.

        Note: This only creates the service object. You MUST call initialize()
        before using any methods or accessing authenticated properties.
        """
        self._config_path = config_path
        self._state_path = state_path
        self._legacy_path = legacy_path

        self._auth_state = AuthState.UNINITIALIZED
        self._config: Optional[dict[str, Any]] = None
        self._persisted: Optional[PersistedState] = None
        self._token_manager: Optional[TokenManager] = None
        self._token_expires_at: Optional[datetime] = None
        self._initialized = False

    # =========================================================================
    # State Guards
    # =========================================================================

    def _require_initialized(self) -> None:
        """Raise if service is not initialized.

        Call this at the start of any method that requires initialization.
        """
        if self._auth_state in (AuthState.UNINITIALIZED, AuthState.LOADING):
            raise NotInitializedError(
                f"FirebaseAuthService.initialize() must be called before using this method. "
                f"Current state: {self._auth_state.name}"
            )

    def _require_authenticated(self) -> None:
        """Raise if service is not authenticated.

        Call this at the start of any method that requires authentication.
        """
        self._require_initialized()
        if self._auth_state != AuthState.AUTHENTICATED:
            raise FirebaseAuthServiceError(
                f"Authentication required. Current state: {self._auth_state.name}"
            )

    # =========================================================================
    # Read-Only Properties (No Side Effects!)
    # =========================================================================

    @property
    def state(self) -> AuthState:
        """Get current authentication state."""
        return self._auth_state

    @property
    def is_initialized(self) -> bool:
        """Check if service has been initialized."""
        return self._initialized

    @property
    def server_id(self) -> str:
        """Get the server ID used for Firebase registration.

        IMPORTANT: This property is read-only and never causes side effects.
        Unlike the old implementation, this will raise if not initialized.
        """
        self._require_initialized()
        if self._persisted is None:
            raise FirebaseAuthServiceError("State not loaded")
        return self._persisted.server_id

    @property
    def user_id(self) -> Optional[str]:
        """Get the current Firebase user ID, or None if not authenticated."""
        if self._persisted is None:
            return None
        return self._persisted.user_id

    @property
    def email(self) -> Optional[str]:
        """Get the current user email, or None if not authenticated."""
        if self._persisted is None:
            return None
        return self._persisted.email

    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return (
            self._auth_state == AuthState.AUTHENTICATED and
            self._persisted is not None and
            self._persisted.has_credentials
        )

    @property
    def id_token(self) -> Optional[str]:
        """Get the current ID token, or None if not authenticated."""
        if self._persisted is None:
            return None
        return self._persisted.id_token

    @property
    def project_id(self) -> Optional[str]:
        """Get the Firebase project ID from config."""
        if self._config is None:
            return None
        return self._config.get("projectId")

    # =========================================================================
    # Initialization
    # =========================================================================

    def _load_config(self) -> bool:
        """Load Firebase configuration from file."""
        if not self._config_path.exists():
            logger.warning("Firebase config not found at %s", self._config_path)
            return False

        try:
            with open(self._config_path, "r") as f:
                self._config = json.load(f)
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load Firebase config: %s", e)
            return False

    def _load_or_create_state(self) -> PersistedState:
        """Load existing state or create new one.

        This method ensures we never lose user_id by:
        1. First attempting to load existing state
        2. Then attempting legacy migration
        3. Only creating new state if nothing exists

        Returns:
            PersistedState (never None - creates new if needed)
        """
        # Try loading existing state
        existing = PersistedState.load(self._state_path)
        if existing:
            logger.debug("Loaded existing state for server: %s", existing.server_id)
            return existing

        # Try legacy migration
        legacy = LegacyDeviceInfo.load(self._legacy_path)
        if legacy:
            logger.info("Migrating legacy device_info.json to server_info.json")
            new_state = legacy.to_persisted_state()
            # Save in new format
            if new_state.save(self._state_path):
                # Remove legacy file
                try:
                    self._legacy_path.unlink()
                    logger.info("Removed legacy device_info.json")
                except OSError:
                    pass
            return new_state

        # Generate new server ID
        server_id = self._generate_server_id()
        new_state = PersistedState(server_id=server_id)
        new_state.save(self._state_path)
        logger.info("Created new server with ID: %s", server_id)
        return new_state

    def _generate_server_id(self) -> str:
        """Generate a unique server ID based on machine identifier.

        Uses a stable machine identifier to ensure the same physical machine
        always gets the same server ID, even if server_info.json is deleted.
        """
        machine_id = self._get_machine_identifier()
        # Create a UUID-like hash for consistency with existing IDs
        hash_hex = hashlib.sha256(machine_id.encode()).hexdigest()[:32]
        # Format as UUID-like string for readability
        return f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"

    def _get_machine_identifier(self) -> str:
        """Get a stable machine identifier."""
        system = platform.system()

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
                                return parts[3]

            elif system == "Linux":
                machine_id_path = Path("/etc/machine-id")
                if machine_id_path.exists():
                    return machine_id_path.read_text().strip()
                dbus_path = Path("/var/lib/dbus/machine-id")
                if dbus_path.exists():
                    return dbus_path.read_text().strip()

            elif system == "Windows":
                result = subprocess.run(
                    ["wmic", "csproduct", "get", "uuid"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) >= 2:
                        return lines[1].strip()

        except (subprocess.SubprocessError, OSError) as e:
            logger.debug("Could not get hardware ID: %s", e)

        # Fallback: use hostname + platform info
        fallback = f"{platform.node()}-{platform.machine()}-{platform.system()}"
        logger.warning("Using fallback machine identifier: %s", fallback)
        return fallback

    async def initialize(self) -> bool:
        """Initialize Firebase auth service.

        This method MUST be called before using any other methods.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        self._auth_state = AuthState.LOADING

        # Load configuration
        if not self._load_config():
            logger.info("Firebase not configured - remote access disabled")
            self._auth_state = AuthState.ERROR
            return False

        # Load or create state (never loses user_id!)
        self._persisted = self._load_or_create_state()

        # Create token manager
        self._token_manager = TokenManager(self._config)
        self._token_manager.start_background_key_fetch()

        # Try to restore authentication
        if self._persisted.has_credentials:
            await self._try_restore_authentication()
        else:
            self._auth_state = AuthState.LOADED

        self._initialized = True
        logger.info("Firebase auth service initialized (token verification mode)")
        return True

    async def _try_restore_authentication(self) -> None:
        """Try to restore authentication from saved tokens."""
        if not self._persisted or not self._token_manager:
            self._auth_state = AuthState.LOADED
            return

        id_token = self._persisted.id_token
        if id_token:
            # Verify saved ID token
            token_info = await self._token_manager.verify_id_token(id_token)
            if token_info:
                self._token_expires_at = token_info.expires_at
                self._auth_state = AuthState.AUTHENTICATED
                logger.info("Restored authentication from saved ID token")
                return

            # Token invalid/expired - try refresh if available
            if self._persisted.has_refresh_capability:
                logger.info("ID token expired, attempting refresh...")
                if await self._refresh_token():
                    return

        elif self._persisted.has_refresh_capability:
            # No ID token but have refresh token
            logger.info("No ID token, attempting refresh...")
            if await self._refresh_token():
                return

        # Could not restore - clear token state but preserve ownership
        logger.warning("Could not restore authentication, clearing tokens (preserving ownership)")
        self._update_state(self._persisted.without_id_token())
        self._auth_state = AuthState.LOADED

    async def _refresh_token(self) -> bool:
        """Attempt to refresh the ID token.

        Returns:
            True if refresh successful and state updated
        """
        if not self._persisted or not self._token_manager:
            return False

        refresh_token = self._persisted.refresh_token
        if not refresh_token:
            return False

        result = await self._token_manager.refresh_id_token(refresh_token)
        if result.success:
            self._update_state(self._persisted.with_credentials(
                id_token=result.id_token,
                refresh_token=result.refresh_token or refresh_token,
            ))
            self._token_expires_at = result.expires_at
            self._auth_state = AuthState.AUTHENTICATED
            logger.info("Authentication restored via refresh token")
            return True

        return False

    # =========================================================================
    # State Management (Explicit Methods Only)
    # =========================================================================

    def _update_state(self, new_state: PersistedState) -> None:
        """Update persisted state and save to disk.

        This is the ONLY way to modify state. All state changes go through here.
        """
        self._persisted = new_state
        new_state.save(self._state_path)

    # =========================================================================
    # Authentication
    # =========================================================================

    async def authenticate_with_token(
        self,
        id_token: str,
        refresh_token: Optional[str] = None,
    ) -> bool:
        """Authenticate using an ID token from the app.

        Args:
            id_token: Firebase ID token from the app
            refresh_token: Firebase refresh token (optional, for permanent auth)

        Returns:
            True if authentication successful, False otherwise
        """
        self._require_initialized()

        if not self._token_manager:
            return False

        token_info = await self._token_manager.verify_id_token(id_token)
        if not token_info:
            return False

        # Update state with new credentials (preserves server_id!)
        self._update_state(self._persisted.with_credentials(
            user_id=token_info.user_id,
            email=token_info.email,
            id_token=id_token,
            refresh_token=refresh_token,
        ))

        self._token_expires_at = token_info.expires_at
        self._auth_state = AuthState.AUTHENTICATED

        logger.info("Authenticated user: %s", token_info.user_id)
        return True

    async def verify_id_token(self, id_token: str) -> Optional[dict[str, Any]]:
        """Verify a Firebase ID token and return the decoded payload.

        Returns:
            Dict with user info, or None if invalid
        """
        self._require_initialized()

        if not self._token_manager:
            return None

        token_info = await self._token_manager.verify_id_token(id_token)
        if not token_info:
            return None

        return {
            "user_id": token_info.user_id,
            "email": token_info.email,
            "email_verified": token_info.email_verified,
            "name": token_info.name,
            "picture": token_info.picture,
        }

    async def refresh_id_token(self) -> bool:
        """Refresh ID token using stored refresh token.

        Returns:
            True if refresh successful, False otherwise
        """
        self._require_initialized()
        return await self._refresh_token()

    async def ensure_valid_token(self) -> bool:
        """Ensure Firebase token is valid, refreshing if needed.

        Returns:
            True if token is valid (or was successfully refreshed)
        """
        self._require_initialized()

        if not self.is_authenticated:
            logger.warning("Not authenticated - cannot ensure valid token")
            return False

        # Can't refresh without refresh_token
        if not self._persisted.has_refresh_capability:
            return True

        # Check if token is expiring soon
        if self._token_expires_at:
            remaining = (self._token_expires_at - datetime.now(timezone.utc)).total_seconds()
            if remaining < TOKEN_REFRESH_THRESHOLD_SECONDS:
                logger.info("Token expiring in %.0fs, refreshing proactively...", remaining)
                return await self._refresh_token()

        return True

    # =========================================================================
    # Device Registration
    # =========================================================================

    async def register_device(
        self,
        tunnel_url: Optional[str] = None,
        local_url: Optional[str] = None,
    ) -> bool:
        """Register this server device to Firestore.

        Args:
            tunnel_url: Cloudflare Tunnel URL (if available)
            local_url: Local network URL (http://ip:port)

        Returns:
            True if registration successful, False otherwise
        """
        self._require_authenticated()

        if not await self.ensure_valid_token():
            logger.error("Token validation failed - cannot register device")
            return False

        return await device_registration.register_device(
            project_id=self.project_id,
            user_id=self.user_id,
            server_id=self.server_id,
            id_token=self.id_token,
            tunnel_url=tunnel_url,
            local_url=local_url,
        )

    async def update_tunnel_url(self, tunnel_url: str) -> bool:
        """Update the Tunnel URL for this device in Firestore."""
        return await self.register_device(tunnel_url)

    async def heartbeat(self) -> bool:
        """Update lastSeen timestamp for this device.

        Returns:
            True if update successful, False otherwise
        """
        self._require_authenticated()

        if not await self.ensure_valid_token():
            logger.error("Token validation failed during heartbeat")
            return False

        success, device_deleted = await device_registration.heartbeat(
            project_id=self.project_id,
            user_id=self.user_id,
            server_id=self.server_id,
            id_token=self.id_token,
        )

        if device_deleted:
            # Device was removed from Firebase - clear auth state
            self._update_state(self._persisted.without_credentials())
            self._auth_state = AuthState.LOADED
            return False

        return success

    # =========================================================================
    # Sign Out / Clear Auth
    # =========================================================================

    async def sign_out(self) -> None:
        """Sign out and clear current session."""
        self._require_initialized()

        self._update_state(self._persisted.without_credentials())
        self._token_expires_at = None
        self._auth_state = AuthState.LOADED
        logger.info("Signed out from Firebase session")

    async def clear_auth(self) -> bool:
        """Clear all authentication data and remove from Firebase.

        Returns:
            True if successful, False otherwise
        """
        self._require_initialized()

        # Try to remove server from Firestore first
        if self.is_authenticated:
            try:
                await device_registration.delete_device(
                    project_id=self.project_id,
                    user_id=self.user_id,
                    server_id=self.server_id,
                    id_token=self.id_token,
                )
            except Exception as e:
                logger.warning("Failed to remove device from Firebase: %s", e)

        # Clear all auth state
        self._update_state(self._persisted.without_credentials())
        self._token_expires_at = None
        self._auth_state = AuthState.LOADED
        logger.info("Authentication cleared")
        return True

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get current authentication status."""
        status: dict[str, Any] = {
            "initialized": self._initialized,
            "state": self._auth_state.name,
            "authenticated": self.is_authenticated,
            "user_id": self.user_id,
            "email": self.email,
            "server_id": self._persisted.server_id if self._persisted else None,
            "server_name": device_registration.get_device_name() if self._persisted else None,
        }

        # Add token expiration info if available
        if self._token_expires_at:
            remaining = (self._token_expires_at - datetime.now(timezone.utc)).total_seconds()
            status["token_expires_at"] = self._token_expires_at.isoformat()
            status["token_expires_in_seconds"] = max(0, int(remaining))
            status["token_needs_refresh"] = remaining < TOKEN_REFRESH_THRESHOLD_SECONDS

        return status
