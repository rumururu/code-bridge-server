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
        self._current_id_token: Optional[str] = None
        self._device_id: Optional[str] = None
        self._public_keys: Optional[dict] = None
        self._keys_fetched_at: Optional[datetime] = None
        self._initialized = False

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
        """Load device info (device_id) from file."""
        if DEVICE_INFO_PATH.exists():
            try:
                with open(DEVICE_INFO_PATH, "r") as f:
                    data = json.load(f)
                    self._device_id = data.get("device_id")
            except Exception as e:
                logger.error(f"Failed to load device info: {e}")

    def _save_device_info(self) -> None:
        """Save device info to file."""
        try:
            with open(DEVICE_INFO_PATH, "w") as f:
                json.dump({"device_id": self._device_id}, f)
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

        # Pre-fetch Google public keys for token verification
        await self._fetch_public_keys()

        self._initialized = True
        logger.info("Firebase auth service initialized (token verification mode)")
        return True

    async def _fetch_public_keys(self) -> bool:
        """Fetch Google's public keys for JWT verification.

        Keys are cached and refreshed every hour.
        """
        # Check if we have recent keys (less than 1 hour old)
        if self._public_keys and self._keys_fetched_at:
            age = (datetime.now(timezone.utc) - self._keys_fetched_at).total_seconds()
            if age < 3600:  # 1 hour
                return True

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(GOOGLE_CERTS_URL)
                if response.status_code == 200:
                    self._public_keys = response.json()
                    self._keys_fetched_at = datetime.now(timezone.utc)
                    logger.debug("Fetched Google public keys for token verification")
                    return True
                else:
                    logger.warning(f"Failed to fetch public keys: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Error fetching public keys: {e}")
            return False

    async def verify_id_token(self, id_token: str) -> Optional[dict]:
        """Verify a Firebase ID token and return the decoded payload.

        This method verifies the token using Google's public keys without
        needing any client secrets.

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

        # Simple token verification using Firebase's token info endpoint
        # This is the easiest way without importing heavy JWT libraries
        try:
            async with httpx.AsyncClient() as client:
                # Use Google's token info endpoint for verification
                # This validates the token and returns the payload
                response = await client.get(
                    f"https://www.googleapis.com/oauth2/v3/tokeninfo",
                    params={"id_token": id_token}
                )

                if response.status_code != 200:
                    logger.warning(f"Token verification failed: {response.text}")
                    return None

                payload = response.json()

                # Verify the audience matches our project
                aud = payload.get("aud")
                if aud != project_id:
                    logger.warning(f"Token audience mismatch: {aud} != {project_id}")
                    return None

                # Verify the issuer
                iss = payload.get("iss")
                expected_issuers = [
                    f"https://securetoken.google.com/{project_id}",
                    "accounts.google.com",
                    "https://accounts.google.com"
                ]
                if iss not in expected_issuers:
                    logger.warning(f"Invalid token issuer: {iss}")
                    return None

                # Extract user info
                user_info = {
                    "user_id": payload.get("sub"),
                    "email": payload.get("email"),
                    "email_verified": payload.get("email_verified") == "true",
                    "name": payload.get("name"),
                    "picture": payload.get("picture"),
                }

                return user_info

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    async def authenticate_with_token(self, id_token: str) -> bool:
        """Authenticate using an ID token from the app.

        This is the main method called when the app sends its Firebase ID token
        to the server. The server verifies the token and stores the user info
        for subsequent Firestore operations.

        Args:
            id_token: Firebase ID token from the app

        Returns:
            True if authentication successful, False otherwise
        """
        user_info = await self.verify_id_token(id_token)
        if not user_info:
            return False

        self._current_user_id = user_info.get("user_id")
        self._current_id_token = id_token

        logger.info(f"Authenticated user: {self._current_user_id}")
        return True

    async def register_device(self, tunnel_url: Optional[str] = None) -> bool:
        """Register this server device to Firestore.

        Requires prior authentication via authenticate_with_token().

        Args:
            tunnel_url: Cloudflare Tunnel URL (if available)

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
                    params={"updateMask.fieldPaths": ["type", "name", "tunnelUrl", "lastSeen"]},
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

                response = await client.patch(
                    url,
                    json={
                        "fields": {
                            "lastSeen": {"timestampValue": datetime.now(timezone.utc).isoformat()},
                        }
                    },
                    headers={"Authorization": f"Bearer {self._current_id_token}"},
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
            "device_id": self._device_id,
            "device_name": self._get_device_name() if self._device_id else None,
        }

    async def sign_out(self) -> None:
        """Sign out and clear current session."""
        self._current_user_id = None
        self._current_id_token = None
        logger.info("Signed out from Firebase session")

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
