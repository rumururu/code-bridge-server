"""Firebase Token Verification for Code Bridge Server.

Option 2 Implementation: App-only authentication, server token verification.
- App handles Google/Apple Sign-In
- App sends Firebase ID Token to server
- Server verifies token using Firebase public keys (no secret needed)
- Server uses verified user ID for Firestore operations
"""

import asyncio
import json
import logging
import platform
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import jwt
from cryptography.x509 import load_pem_x509_certificate

logger = logging.getLogger(__name__)

# Firebase configuration - public config only (no secrets)
FIREBASE_CONFIG_PATH = Path(__file__).parent / "firebase_config.json"
SERVER_INFO_PATH = Path(__file__).parent / "server_info.json"
LEGACY_DEVICE_INFO_PATH = Path(__file__).parent / "device_info.json"  # For migration

# Google's public keys for JWT verification
GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"


class FirebaseAuthService:
    """Manages Firebase token verification and Firestore operations.

    This service verifies ID tokens sent from the app and uses the verified
    user information for device registration and Firestore access.
    """

    # Refresh token 10 minutes before expiration
    TOKEN_REFRESH_THRESHOLD_SECONDS = 600

    def __init__(self):
        """Initialize Firebase auth service."""
        self._config: Optional[dict] = None
        self._current_user_id: Optional[str] = None
        self._current_email: Optional[str] = None
        self._current_id_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._auth_mode: str = "id_token"  # "id_token" or "refresh_token"
        self._server_id: Optional[str] = None
        self._public_keys: Optional[dict] = None
        self._keys_fetched_at: Optional[datetime] = None
        self._initialized = False
        self._keys_refresh_task: Optional[asyncio.Task] = None
        self._token_expires_at: Optional[datetime] = None

    def _load_config(self) -> bool:
        """Load Firebase configuration from file.

        Only needs public config (apiKey, projectId) - no secrets required.
        """
        if not FIREBASE_CONFIG_PATH.exists():
            logger.warning(f"Firebase config not found at {FIREBASE_CONFIG_PATH}")
            return False

        try:
            with open(FIREBASE_CONFIG_PATH, "r") as f:
                self._config = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Failed to load Firebase config: {e}")
            return False

    def _load_server_info(self) -> None:
        """Load server info from file with legacy migration support."""
        # Try new format first
        if SERVER_INFO_PATH.exists():
            try:
                with open(SERVER_INFO_PATH, "r") as f:
                    data = json.load(f)
                    self._server_id = data.get("server_id")
                    self._current_user_id = data.get("user_id")
                    self._current_email = data.get("email")
                    self._current_id_token = data.get("id_token")
                    self._refresh_token = data.get("refresh_token")
                    self._auth_mode = data.get("auth_mode", "id_token")
                return
            except Exception as e:
                logger.error(f"Failed to load server info: {e}")

        # Migrate from legacy device_info.json
        if LEGACY_DEVICE_INFO_PATH.exists():
            try:
                with open(LEGACY_DEVICE_INFO_PATH, "r") as f:
                    data = json.load(f)
                    self._server_id = data.get("device_id")  # Legacy key
                    self._current_user_id = data.get("user_id")
                    self._current_email = data.get("email")
                    self._current_id_token = data.get("id_token")
                    self._refresh_token = data.get("refresh_token")
                    self._auth_mode = data.get("auth_mode", "id_token")
                # Save in new format
                self._save_server_info()
                # Remove legacy file
                LEGACY_DEVICE_INFO_PATH.unlink()
                logger.info("Migrated device_info.json to server_info.json")
            except Exception as e:
                logger.error(f"Failed to migrate legacy device info: {e}")

    def _save_server_info(self) -> None:
        """Save server info to file."""
        try:
            data = {"server_id": self._server_id}
            if self._current_user_id:
                data["user_id"] = self._current_user_id
            if self._current_email:
                data["email"] = self._current_email
            if self._auth_mode:
                data["auth_mode"] = self._auth_mode
            if self._auth_mode == "refresh_token" and self._refresh_token:
                data["refresh_token"] = self._refresh_token
            elif self._auth_mode == "id_token" and self._current_id_token:
                data["id_token"] = self._current_id_token
            with open(SERVER_INFO_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save server info: {e}")

    def _get_server_id(self) -> str:
        """Get or generate a unique server ID."""
        if self._server_id:
            return self._server_id

        # Generate a new server ID
        self._server_id = str(uuid.uuid4())
        self._save_server_info()
        return self._server_id

    def _get_device_name(self) -> str:
        """Get a human-readable device name."""
        hostname = platform.node()
        system = platform.system()
        return f"{hostname} ({system})"

    async def initialize(self) -> bool:
        """Initialize Firebase auth service.

        Non-blocking initialization - public keys are fetched in background.
        Token verification will wait for keys if needed.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        if not self._load_config():
            logger.info("Firebase not configured - remote access disabled")
            return False

        self._load_server_info()
        self._get_server_id()  # Ensure server ID exists

        # Start background fetch of Google public keys (non-blocking)
        self._start_background_key_fetch()

        # Restore authentication from saved tokens
        if self._current_user_id:
            if self._current_id_token:
                # Try to verify saved ID token (will wait for keys if needed)
                user_info = await self.verify_id_token(self._current_id_token)
                if user_info:
                    # Extract token expiration for proactive refresh
                    self._extract_token_expiration(self._current_id_token)
                    logger.info(f"Restored authentication from saved ID token ({self._auth_mode} mode)")
                elif self._auth_mode == "refresh_token" and self._refresh_token:
                    # ID token expired, try refresh
                    logger.info("ID token expired, attempting refresh...")
                    if await self.refresh_id_token():
                        logger.info(f"Restored authentication via refresh token")
                    else:
                        logger.warning("Refresh token invalid, clearing auth")
                        self._clear_auth_state()
                else:
                    logger.info("Saved ID token expired, clearing (id_token mode)")
                    self._clear_auth_state()
            elif self._auth_mode == "refresh_token" and self._refresh_token:
                # No ID token but have refresh token
                logger.info("No ID token, attempting refresh...")
                if await self.refresh_id_token():
                    logger.info(f"Restored authentication via refresh token")
                else:
                    logger.warning("Refresh token invalid, clearing auth")
                    self._clear_auth_state()

        self._initialized = True
        logger.info("Firebase auth service initialized (token verification mode)")
        return True

    def _start_background_key_fetch(self) -> None:
        """Start background task to fetch public keys."""
        if self._keys_refresh_task is None or self._keys_refresh_task.done():
            self._keys_refresh_task = asyncio.create_task(self._fetch_public_keys())

    def _clear_auth_state(self) -> None:
        """Clear authentication state (internal use)."""
        self._current_user_id = None
        self._current_email = None
        self._current_id_token = None
        self._refresh_token = None
        self._auth_mode = "id_token"
        self._token_expires_at = None

    def _extract_token_expiration(self, id_token: str) -> None:
        """Extract and store token expiration from ID token.

        Args:
            id_token: Firebase ID token to extract expiration from
        """
        try:
            unverified = jwt.decode(id_token, options={"verify_signature": False})
            exp = unverified.get("exp")
            if exp:
                self._token_expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
                remaining = (self._token_expires_at - datetime.now(timezone.utc)).total_seconds()
                logger.debug(f"Token expires at: {self._token_expires_at} ({remaining:.0f}s remaining)")
        except Exception as e:
            logger.debug(f"Could not extract token expiration: {e}")

    async def _fetch_public_keys(self) -> bool:
        """Fetch Google's public keys for JWT verification.

        Keys are cached and refreshed every hour. Has timeout to prevent blocking.
        """
        # Check if we have recent keys (less than 1 hour old)
        if self._public_keys and self._keys_fetched_at:
            age = (datetime.now(timezone.utc) - self._keys_fetched_at).total_seconds()
            if age < 3600:  # 1 hour
                return True

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(GOOGLE_CERTS_URL)
                if response.status_code == 200:
                    self._public_keys = response.json()
                    self._keys_fetched_at = datetime.now(timezone.utc)
                    logger.debug("Fetched Google public keys for token verification")
                    return True
                else:
                    logger.warning(f"Failed to fetch public keys: {response.status_code}")
                    return False
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching public keys (will retry)")
            return False
        except Exception as e:
            logger.error(f"Error fetching public keys: {e}")
            return False

    async def _ensure_public_keys(self) -> bool:
        """Ensure public keys are available, waiting for background fetch if needed."""
        # If we already have keys, return immediately
        if self._public_keys:
            # Trigger background refresh if stale
            if self._keys_fetched_at:
                age = (datetime.now(timezone.utc) - self._keys_fetched_at).total_seconds()
                if age >= 3600:  # 1 hour
                    self._start_background_key_fetch()
            return True

        # Wait for background fetch to complete (with timeout)
        if self._keys_refresh_task and not self._keys_refresh_task.done():
            try:
                await asyncio.wait_for(self._keys_refresh_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for public keys")
                return False
            return self._public_keys is not None

        # No background task, fetch now
        return await self._fetch_public_keys()

    async def verify_id_token(self, id_token: str) -> Optional[dict]:
        """Verify a Firebase ID token and return the decoded payload.

        This method verifies the token using Google's public keys (X.509 certificates)
        with proper JWT signature verification using PyJWT.

        Args:
            id_token: Firebase ID token from the app

        Returns:
            Decoded token payload with user info, or None if invalid
        """
        if not self._config:
            logger.error("Firebase not configured")
            return None

        project_id = self._config.get("projectId")
        if not project_id:
            logger.error("Missing projectId in Firebase config")
            return None

        # Ensure Google's public keys are available (waits for background fetch if needed)
        if not await self._ensure_public_keys():
            logger.error("Could not obtain Google public keys")
            return None

        try:
            # Decode header to get the key ID (kid)
            unverified_header = jwt.get_unverified_header(id_token)
            kid = unverified_header.get("kid")

            if not kid or kid not in self._public_keys:
                logger.warning(f"Unknown key ID in token: {kid}")
                return None

            # Get the certificate for this key ID and extract public key
            cert_pem = self._public_keys[kid]
            cert = load_pem_x509_certificate(cert_pem.encode())
            public_key = cert.public_key()

            # Verify and decode the token
            payload = jwt.decode(
                id_token,
                public_key,
                algorithms=["RS256"],
                audience=project_id,
                issuer=f"https://securetoken.google.com/{project_id}",
            )

            # Extract user info
            user_info = {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "email_verified": payload.get("email_verified", False),
                "name": payload.get("name"),
                "picture": payload.get("picture"),
            }

            logger.info(f"Token verified for user: {user_info.get('email')}")
            return user_info

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidAudienceError:
            logger.warning("Token audience mismatch")
            return None
        except jwt.InvalidIssuerError:
            logger.warning("Token issuer mismatch")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    async def refresh_id_token(self) -> bool:
        """Refresh ID token using stored refresh token.

        Uses Firebase Auth REST API to exchange refresh token for new ID token.

        Returns:
            True if refresh successful, False otherwise
        """
        if not self._refresh_token or not self._config:
            logger.warning("No refresh token available")
            return False

        api_key = self._config.get("apiKey")
        if not api_key:
            logger.error("Missing apiKey in Firebase config")
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://securetoken.googleapis.com/v1/token?key={api_key}",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self._refresh_token,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    self._current_id_token = data.get("id_token")
                    # Refresh token might be rotated
                    new_refresh_token = data.get("refresh_token")
                    if new_refresh_token:
                        self._refresh_token = new_refresh_token

                    # Extract and store token expiration from expires_in field
                    expires_in = data.get("expires_in")
                    if expires_in:
                        try:
                            self._token_expires_at = datetime.now(timezone.utc) + \
                                timedelta(seconds=int(expires_in))
                            logger.debug(f"Token expires at: {self._token_expires_at}")
                        except (ValueError, TypeError):
                            # Default to 1 hour if expires_in is invalid
                            self._token_expires_at = datetime.now(timezone.utc) + \
                                timedelta(hours=1)

                    self._save_server_info()
                    logger.info("ID token refreshed successfully")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    async def ensure_valid_token(self) -> bool:
        """Ensure Firebase token is valid, refreshing if needed.

        This method should be called before any Firebase API call to ensure
        the token is valid and hasn't expired. For refresh_token mode, it
        proactively refreshes the token if it's close to expiring.

        Returns:
            True if token is valid (or was successfully refreshed), False otherwise
        """
        if not self.is_authenticated:
            logger.warning("Not authenticated - cannot ensure valid token")
            return False

        # For id_token mode, we can't refresh - just return current state
        if self._auth_mode != "refresh_token" or not self._refresh_token:
            return True

        # Check if token is expiring soon
        if self._token_expires_at:
            remaining = (self._token_expires_at - datetime.now(timezone.utc)).total_seconds()
            if remaining < self.TOKEN_REFRESH_THRESHOLD_SECONDS:
                logger.info(f"Token expiring in {remaining:.0f}s, refreshing proactively...")
                if await self.refresh_id_token():
                    logger.info("Token refreshed successfully before expiration")
                    return True
                else:
                    logger.error("Failed to refresh token before expiration")
                    return False
        else:
            # No expiration info - try to extract from current token
            if self._current_id_token:
                try:
                    # Decode without verification to get expiration
                    unverified = jwt.decode(
                        self._current_id_token,
                        options={"verify_signature": False}
                    )
                    exp = unverified.get("exp")
                    if exp:
                        self._token_expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
                        remaining = (self._token_expires_at - datetime.now(timezone.utc)).total_seconds()
                        if remaining < self.TOKEN_REFRESH_THRESHOLD_SECONDS:
                            logger.info(f"Token expiring in {remaining:.0f}s, refreshing...")
                            return await self.refresh_id_token()
                except Exception as e:
                    logger.debug(f"Could not decode token for expiration check: {e}")

        return True

    async def authenticate_with_token(
        self, id_token: str, refresh_token: Optional[str] = None, auth_mode: str = "refresh_token"
    ) -> bool:
        """Authenticate using an ID token from the app.

        This is the main method called when the app sends its Firebase ID token
        to the server. The server verifies the token and stores the user info
        for subsequent Firestore operations.

        Args:
            id_token: Firebase ID token from the app
            refresh_token: Firebase refresh token (optional, for refresh_token mode)
            auth_mode: "id_token" (1 hour) or "refresh_token" (permanent)

        Returns:
            True if authentication successful, False otherwise
        """
        user_info = await self.verify_id_token(id_token)
        if not user_info:
            return False

        self._current_user_id = user_info.get("user_id")
        self._current_email = user_info.get("email")
        self._current_id_token = id_token
        self._auth_mode = auth_mode

        # Extract token expiration for proactive refresh
        try:
            unverified = jwt.decode(id_token, options={"verify_signature": False})
            exp = unverified.get("exp")
            if exp:
                self._token_expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
                logger.debug(f"Token expires at: {self._token_expires_at}")
        except Exception as e:
            logger.debug(f"Could not extract token expiration: {e}")

        if auth_mode == "refresh_token" and refresh_token:
            self._refresh_token = refresh_token
            logger.info(f"Authenticated user (refresh_token mode): {self._current_user_id}")
        else:
            self._auth_mode = "id_token"
            logger.info(f"Authenticated user (id_token mode): {self._current_user_id}")

        # Save token to file for persistence across restarts
        self._save_server_info()

        return True

    async def register_device(self, tunnel_url: Optional[str] = None, local_url: Optional[str] = None) -> bool:
        """Register this server device to Firestore.

        Requires prior authentication via authenticate_with_token().

        Args:
            tunnel_url: Cloudflare Tunnel URL (if available)
            local_url: Local network URL (http://ip:port)

        Returns:
            True if registration successful, False otherwise
        """
        if not self._current_user_id or not self._current_id_token or not self._config:
            logger.warning("Not authenticated - cannot register device")
            return False

        # Ensure token is valid before making Firebase API call
        if not await self.ensure_valid_token():
            logger.error("Token validation failed - cannot register device")
            return False

        project_id = self._config.get("projectId")
        if not project_id:
            return False

        server_id = self._get_server_id()
        now = datetime.now(timezone.utc).isoformat()

        device_data = {
            "fields": {
                "type": {"stringValue": "server"},
                "name": {"stringValue": self._get_device_name()},
                "tunnelUrl": {"stringValue": tunnel_url or ""},
                "localUrl": {"stringValue": local_url or ""},
                "lastSeen": {"timestampValue": now},
                "createdAt": {"timestampValue": now},
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                # Use Firestore REST API
                url = (
                    f"https://firestore.googleapis.com/v1/"
                    f"projects/{project_id}/databases/(default)/documents/"
                    f"users/{self._current_user_id}/devices/{server_id}"
                )

                response = await client.patch(
                    url,
                    json=device_data,
                    headers={"Authorization": f"Bearer {self._current_id_token}"},
                    params={"updateMask.fieldPaths": ["type", "name", "tunnelUrl", "localUrl", "lastSeen"]},
                )

                if response.status_code in (200, 201):
                    logger.info(f"Server registered: {server_id}")
                    return True
                else:
                    logger.error(f"Device registration failed: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Device registration error: {e}")
            return False

    async def update_tunnel_url(self, tunnel_url: str) -> bool:
        """Update the Tunnel URL for this device in Firestore.

        Args:
            tunnel_url: New Cloudflare Tunnel URL

        Returns:
            True if update successful, False otherwise
        """
        return await self.register_device(tunnel_url)

    async def heartbeat(self) -> bool:
        """Update lastSeen timestamp for this device.

        Checks if the device document still exists before updating.
        If the document was deleted (e.g., by user from app), clears auth state.
        Proactively refreshes token if it's close to expiring.

        Returns:
            True if update successful, False otherwise
        """
        if not self._current_user_id or not self._current_id_token or not self._config:
            return False

        # Ensure token is valid before making Firebase API call
        if not await self.ensure_valid_token():
            logger.error("Token validation failed during heartbeat")
            return False

        project_id = self._config.get("projectId")
        if not project_id:
            return False

        server_id = self._get_server_id()

        try:
            async with httpx.AsyncClient() as client:
                url = (
                    f"https://firestore.googleapis.com/v1/"
                    f"projects/{project_id}/databases/(default)/documents/"
                    f"users/{self._current_user_id}/devices/{server_id}"
                )
                headers = {"Authorization": f"Bearer {self._current_id_token}"}

                # First, check if document still exists (GET request)
                check_response = await client.get(url, headers=headers)

                if check_response.status_code == 404:
                    # Document was deleted - device removed from Firebase by user
                    logger.warning("Device was removed from Firebase. Re-pairing required.")
                    self._clear_auth_state()
                    self._save_server_info()
                    return False

                if check_response.status_code == 403:
                    # Permission denied - token expired or revoked
                    logger.warning("Firebase auth expired or revoked.")
                    return False

                # Document exists, update lastSeen
                response = await client.patch(
                    url,
                    json={
                        "fields": {
                            "lastSeen": {"timestampValue": datetime.now(timezone.utc).isoformat()},
                        }
                    },
                    headers=headers,
                    params={"updateMask.fieldPaths": ["lastSeen"]},
                )

                return response.status_code in (200, 201)

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get current authentication status."""
        status = {
            "initialized": self._initialized,
            "authenticated": self._current_user_id is not None,
            "user_id": self._current_user_id,
            "email": self._current_email,
            "server_id": self._server_id,
            "server_name": self._get_device_name() if self._server_id else None,
            "auth_mode": self._auth_mode,
        }

        # Add token expiration info if available
        if self._token_expires_at:
            remaining = (self._token_expires_at - datetime.now(timezone.utc)).total_seconds()
            status["token_expires_at"] = self._token_expires_at.isoformat()
            status["token_expires_in_seconds"] = max(0, int(remaining))
            status["token_needs_refresh"] = remaining < self.TOKEN_REFRESH_THRESHOLD_SECONDS

        return status

    async def sign_out(self) -> None:
        """Sign out and clear current session."""
        self._clear_auth_state()
        logger.info("Signed out from Firebase session")

    async def clear_auth(self) -> bool:
        """Clear all authentication data and remove from Firebase.

        Called when app disconnects from server. Removes device from Firestore
        and clears all saved tokens.

        Returns:
            True if successful, False otherwise
        """
        # Try to remove server from Firestore first
        if self._current_user_id and self._current_id_token and self._config:
            try:
                project_id = self._config.get("projectId")
                server_id = self._get_server_id()
                if project_id and server_id:
                    async with httpx.AsyncClient() as client:
                        url = (
                            f"https://firestore.googleapis.com/v1/"
                            f"projects/{project_id}/databases/(default)/documents/"
                            f"users/{self._current_user_id}/devices/{server_id}"
                        )
                        await client.delete(
                            url,
                            headers={"Authorization": f"Bearer {self._current_id_token}"},
                        )
                        logger.info(f"Server removed from Firebase: {server_id}")
            except Exception as e:
                logger.warning(f"Failed to remove device from Firebase: {e}")

        # Clear all auth state
        self._clear_auth_state()
        self._save_server_info()
        logger.info("Authentication cleared")
        return True

    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self._current_user_id is not None and self._current_id_token is not None

    @property
    def server_id(self) -> str:
        """Get the server ID used for Firebase registration."""
        return self._get_server_id()


# Singleton instance
_firebase_auth: Optional[FirebaseAuthService] = None


def get_firebase_auth() -> FirebaseAuthService:
    """Get or create Firebase auth service singleton."""
    global _firebase_auth
    if _firebase_auth is None:
        _firebase_auth = FirebaseAuthService()
    return _firebase_auth
