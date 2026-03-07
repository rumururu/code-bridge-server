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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import jwt
from cryptography.x509 import load_pem_x509_certificate

logger = logging.getLogger(__name__)

# Firebase configuration - public config only (no secrets)
FIREBASE_CONFIG_PATH = Path(__file__).parent / "firebase_config.json"
DEVICE_INFO_PATH = Path(__file__).parent / "device_info.json"

# Google's public keys for JWT verification
GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"


class FirebaseAuthService:
    """Manages Firebase token verification and Firestore operations.

    This service verifies ID tokens sent from the app and uses the verified
    user information for device registration and Firestore access.
    """

    def __init__(self):
        """Initialize Firebase auth service."""
        self._config: Optional[dict] = None
        self._current_user_id: Optional[str] = None
        self._current_email: Optional[str] = None
        self._current_id_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._auth_mode: str = "id_token"  # "id_token" or "refresh_token"
        self._device_id: Optional[str] = None
        self._public_keys: Optional[dict] = None
        self._keys_fetched_at: Optional[datetime] = None
        self._initialized = False
        self._keys_refresh_task: Optional[asyncio.Task] = None

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

    def _load_device_info(self) -> None:
        """Load device info from file."""
        if DEVICE_INFO_PATH.exists():
            try:
                with open(DEVICE_INFO_PATH, "r") as f:
                    data = json.load(f)
                    self._device_id = data.get("device_id")
                    self._current_user_id = data.get("user_id")
                    self._current_email = data.get("email")
                    self._current_id_token = data.get("id_token")
                    self._refresh_token = data.get("refresh_token")
                    self._auth_mode = data.get("auth_mode", "id_token")
            except Exception as e:
                logger.error(f"Failed to load device info: {e}")

    def _save_device_info(self) -> None:
        """Save device info to file."""
        try:
            data = {"device_id": self._device_id}
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
            with open(DEVICE_INFO_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save device info: {e}")

    def _get_device_id(self) -> str:
        """Get or generate a unique device ID."""
        if self._device_id:
            return self._device_id

        # Generate a new device ID
        self._device_id = str(uuid.uuid4())
        self._save_device_info()
        return self._device_id

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

        self._load_device_info()
        self._get_device_id()  # Ensure device ID exists

        # Start background fetch of Google public keys (non-blocking)
        self._start_background_key_fetch()

        # Restore authentication from saved tokens
        if self._current_user_id:
            if self._current_id_token:
                # Try to verify saved ID token (will wait for keys if needed)
                user_info = await self.verify_id_token(self._current_id_token)
                if user_info:
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
                    self._save_device_info()
                    logger.info("ID token refreshed successfully")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

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

        if auth_mode == "refresh_token" and refresh_token:
            self._refresh_token = refresh_token
            logger.info(f"Authenticated user (refresh_token mode): {self._current_user_id}")
        else:
            self._auth_mode = "id_token"
            logger.info(f"Authenticated user (id_token mode): {self._current_user_id}")

        # Save token to file for persistence across restarts
        self._save_device_info()

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

        project_id = self._config.get("projectId")
        if not project_id:
            return False

        device_id = self._get_device_id()
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
                    f"users/{self._current_user_id}/devices/{device_id}"
                )

                response = await client.patch(
                    url,
                    json=device_data,
                    headers={"Authorization": f"Bearer {self._current_id_token}"},
                    params={"updateMask.fieldPaths": ["type", "name", "tunnelUrl", "localUrl", "lastSeen"]},
                )

                if response.status_code in (200, 201):
                    logger.info(f"Device registered: {device_id}")
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

        Returns:
            True if update successful, False otherwise
        """
        if not self._current_user_id or not self._current_id_token or not self._config:
            return False

        project_id = self._config.get("projectId")
        if not project_id:
            return False

        device_id = self._get_device_id()

        try:
            async with httpx.AsyncClient() as client:
                url = (
                    f"https://firestore.googleapis.com/v1/"
                    f"projects/{project_id}/databases/(default)/documents/"
                    f"users/{self._current_user_id}/devices/{device_id}"
                )
                headers = {"Authorization": f"Bearer {self._current_id_token}"}

                # First, check if document still exists (GET request)
                check_response = await client.get(url, headers=headers)

                if check_response.status_code == 404:
                    # Document was deleted - device removed from Firebase by user
                    logger.warning("Device was removed from Firebase. Re-pairing required.")
                    self._clear_auth_state()
                    self._save_device_info()
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
        return {
            "initialized": self._initialized,
            "authenticated": self._current_user_id is not None,
            "user_id": self._current_user_id,
            "email": self._current_email,
            "device_id": self._device_id,
            "device_name": self._get_device_name() if self._device_id else None,
            "auth_mode": self._auth_mode,
        }

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
        # Try to remove device from Firestore first
        if self._current_user_id and self._current_id_token and self._config:
            try:
                project_id = self._config.get("projectId")
                device_id = self._get_device_id()
                if project_id and device_id:
                    async with httpx.AsyncClient() as client:
                        url = (
                            f"https://firestore.googleapis.com/v1/"
                            f"projects/{project_id}/databases/(default)/documents/"
                            f"users/{self._current_user_id}/devices/{device_id}"
                        )
                        await client.delete(
                            url,
                            headers={"Authorization": f"Bearer {self._current_id_token}"},
                        )
                        logger.info(f"Device removed from Firebase: {device_id}")
            except Exception as e:
                logger.warning(f"Failed to remove device from Firebase: {e}")

        # Clear all auth state
        self._clear_auth_state()
        self._save_device_info()
        logger.info("Authentication cleared")
        return True

    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self._current_user_id is not None and self._current_id_token is not None


# Singleton instance
_firebase_auth: Optional[FirebaseAuthService] = None


def get_firebase_auth() -> FirebaseAuthService:
    """Get or create Firebase auth service singleton."""
    global _firebase_auth
    if _firebase_auth is None:
        _firebase_auth = FirebaseAuthService()
    return _firebase_auth
