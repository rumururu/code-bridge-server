"""Firebase token verification and refresh management.

This module handles:
- ID token verification using Google's public keys
- Token refresh using Firebase REST API
- Public key caching and background refresh
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
import jwt
from cryptography.x509 import load_pem_x509_certificate

logger = logging.getLogger(__name__)

# Google's public keys for JWT verification
GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"

# Token refresh threshold (refresh 10 minutes before expiration)
TOKEN_REFRESH_THRESHOLD_SECONDS = 600


@dataclass
class TokenInfo:
    """Information extracted from a verified token."""
    user_id: str
    email: Optional[str]
    email_verified: bool
    name: Optional[str]
    picture: Optional[str]
    expires_at: Optional[datetime]


@dataclass
class RefreshResult:
    """Result of a token refresh operation."""
    success: bool
    id_token: Optional[str] = None
    refresh_token: Optional[str] = None  # May be rotated
    expires_at: Optional[datetime] = None
    error: Optional[str] = None


class TokenManager:
    """Manages Firebase token verification and refresh.

    This class handles:
    - Fetching and caching Google's public keys
    - Verifying Firebase ID tokens
    - Refreshing tokens using Firebase REST API
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize token manager.

        Args:
            config: Firebase configuration dict with projectId and apiKey
        """
        self._config = config
        self._public_keys: Optional[dict[str, str]] = None
        self._keys_fetched_at: Optional[datetime] = None
        self._keys_refresh_task: Optional[asyncio.Task] = None

    @property
    def project_id(self) -> Optional[str]:
        """Get Firebase project ID."""
        return self._config.get("projectId")

    @property
    def api_key(self) -> Optional[str]:
        """Get Firebase API key."""
        return self._config.get("apiKey")

    def start_background_key_fetch(self) -> None:
        """Start background task to fetch public keys."""
        if self._keys_refresh_task is None or self._keys_refresh_task.done():
            self._keys_refresh_task = asyncio.create_task(self._fetch_public_keys())

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
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(GOOGLE_CERTS_URL)
                if response.status_code == 200:
                    self._public_keys = response.json()
                    self._keys_fetched_at = datetime.now(timezone.utc)
                    logger.debug("Fetched Google public keys for token verification")
                    return True
                else:
                    logger.warning("Failed to fetch public keys: %s", response.status_code)
                    return False
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching public keys (will retry)")
            return False
        except httpx.HTTPError as e:
            logger.error("Error fetching public keys: %s", e)
            return False

    async def _ensure_public_keys(self) -> bool:
        """Ensure public keys are available, waiting for background fetch if needed."""
        # If we already have keys, return immediately
        if self._public_keys:
            # Trigger background refresh if stale
            if self._keys_fetched_at:
                age = (datetime.now(timezone.utc) - self._keys_fetched_at).total_seconds()
                if age >= 3600:  # 1 hour
                    self.start_background_key_fetch()
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

    async def verify_id_token(self, id_token: str) -> Optional[TokenInfo]:
        """Verify a Firebase ID token and return extracted info.

        This method verifies the token using Google's public keys (X.509 certificates)
        with proper JWT signature verification using PyJWT.

        Args:
            id_token: Firebase ID token from the app

        Returns:
            TokenInfo if valid, None if invalid or expired
        """
        if not self.project_id:
            logger.error("Missing projectId in Firebase config")
            return None

        # Ensure Google's public keys are available
        if not await self._ensure_public_keys():
            logger.error("Could not obtain Google public keys")
            return None

        try:
            # Decode header to get the key ID (kid)
            unverified_header = jwt.get_unverified_header(id_token)
            kid = unverified_header.get("kid")

            if not kid or kid not in self._public_keys:
                logger.warning("Unknown key ID in token: %s", kid)
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
                audience=self.project_id,
                issuer=f"https://securetoken.google.com/{self.project_id}",
            )

            # Extract expiration
            exp = payload.get("exp")
            expires_at = None
            if exp:
                expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)

            token_info = TokenInfo(
                user_id=payload.get("sub", ""),
                email=payload.get("email"),
                email_verified=payload.get("email_verified", False),
                name=payload.get("name"),
                picture=payload.get("picture"),
                expires_at=expires_at,
            )

            logger.info("Token verified for user: %s", token_info.email)
            return token_info

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
            logger.warning("Invalid token: %s", e)
            return None

    def extract_token_expiration(self, id_token: str) -> Optional[datetime]:
        """Extract expiration timestamp from ID token without verification.

        Used for proactive refresh scheduling.

        Args:
            id_token: Firebase ID token

        Returns:
            Expiration datetime if extractable, None otherwise
        """
        try:
            unverified = jwt.decode(id_token, options={"verify_signature": False})
            exp = unverified.get("exp")
            if exp:
                return datetime.fromtimestamp(exp, tz=timezone.utc)
        except jwt.PyJWTError as e:
            logger.debug("Could not extract token expiration: %s", e)
        return None

    def token_needs_refresh(self, expires_at: Optional[datetime]) -> bool:
        """Check if token needs refresh based on expiration.

        Args:
            expires_at: Token expiration timestamp

        Returns:
            True if token should be refreshed
        """
        if not expires_at:
            return False

        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        return remaining < TOKEN_REFRESH_THRESHOLD_SECONDS

    async def refresh_id_token(self, refresh_token: str) -> RefreshResult:
        """Refresh ID token using stored refresh token.

        Uses Firebase Auth REST API to exchange refresh token for new ID token.

        Args:
            refresh_token: Firebase refresh token

        Returns:
            RefreshResult with new tokens if successful
        """
        if not self.api_key:
            logger.error("Missing apiKey in Firebase config")
            return RefreshResult(success=False, error="Missing apiKey")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://securetoken.googleapis.com/v1/token?key={self.api_key}",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    new_id_token = data.get("id_token")
                    new_refresh_token = data.get("refresh_token")

                    # Calculate expiration from expires_in field
                    expires_at = None
                    expires_in = data.get("expires_in")
                    if expires_in:
                        try:
                            expires_at = datetime.now(timezone.utc) + \
                                timedelta(seconds=int(expires_in))
                        except (ValueError, TypeError):
                            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

                    logger.info("ID token refreshed successfully")
                    return RefreshResult(
                        success=True,
                        id_token=new_id_token,
                        refresh_token=new_refresh_token,
                        expires_at=expires_at,
                    )
                else:
                    error_msg = f"Token refresh failed: {response.text}"
                    logger.error(error_msg)
                    return RefreshResult(success=False, error=error_msg)

        except httpx.HTTPError as e:
            error_msg = f"Token refresh error: {e}"
            logger.error(error_msg)
            return RefreshResult(success=False, error=error_msg)
