"""Multi-account Firebase pairing management.

This module manages multiple Firebase accounts paired with a single server.
Each paired account has its own credentials and can receive server URL updates.

Storage: CODEBRIDGE_APP_SUPPORT_DIR/paired_accounts.json in packaged desktop
mode, otherwise ~/.code-bridge/paired_accounts.json.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from core.runtime_paths import runtime_path

from . import device_registration
from .token_manager import TOKEN_REFRESH_THRESHOLD_SECONDS, TokenManager

logger = logging.getLogger(__name__)

# Default storage path
PAIRED_ACCOUNTS_PATH = runtime_path("paired_accounts.json", Path.home() / ".code-bridge" / "paired_accounts.json")


@dataclass
class PairedAccount:
    """A Firebase account paired with this server."""

    user_id: str
    email: Optional[str] = None
    id_token: Optional[str] = None
    refresh_token: Optional[str] = None
    paired_at: Optional[str] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {"user_id": self.user_id}
        if self.email:
            data["email"] = self.email
        if self.id_token:
            data["id_token"] = self.id_token
        if self.refresh_token:
            data["refresh_token"] = self.refresh_token
        if self.paired_at:
            data["paired_at"] = self.paired_at
        if self.last_updated:
            data["last_updated"] = self.last_updated
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional[PairedAccount]:
        """Create from dictionary."""
        user_id = data.get("user_id")
        if not user_id:
            return None
        return cls(
            user_id=user_id,
            email=data.get("email"),
            id_token=data.get("id_token"),
            refresh_token=data.get("refresh_token"),
            paired_at=data.get("paired_at"),
            last_updated=data.get("last_updated"),
        )

    def with_updated_token(
        self,
        id_token: str,
        refresh_token: Optional[str] = None,
    ) -> PairedAccount:
        """Create a new instance with updated token."""
        return PairedAccount(
            user_id=self.user_id,
            email=self.email,
            id_token=id_token,
            refresh_token=refresh_token if refresh_token else self.refresh_token,
            paired_at=self.paired_at,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )


class PairedAccountsManager:
    """Manages multiple Firebase accounts paired with this server.

    Responsibilities:
    - Store and retrieve paired account credentials
    - Refresh tokens for each account
    - Update Firestore URLs for all accounts when server URL changes
    """

    def __init__(
        self,
        storage_path: Path = PAIRED_ACCOUNTS_PATH,
        firebase_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the manager.

        Args:
            storage_path: Path to store paired accounts JSON
            firebase_config: Firebase config for token operations
        """
        self._storage_path = storage_path
        self._firebase_config = firebase_config
        self._accounts: dict[str, PairedAccount] = {}
        self._token_helper: Optional[TokenManager] = None
        self._load()

    @property
    def _tokens(self) -> TokenManager:
        """Lazy TokenManager, used only for its two pure token helpers.

        ``TokenManager.__init__`` does no I/O and starts no tasks, so building
        one on demand is cheap. We never call its network methods here.
        """
        if self._token_helper is None:
            self._token_helper = TokenManager(self._firebase_config or {})
        return self._token_helper

    def _load(self) -> None:
        """Load paired accounts from disk."""
        if not self._storage_path.exists():
            self._accounts = {}
            return

        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)

            accounts_data = data.get("accounts", [])
            self._accounts = {}
            for acc_data in accounts_data:
                account = PairedAccount.from_dict(acc_data)
                if account:
                    self._accounts[account.user_id] = account

            logger.debug("Loaded %d paired accounts", len(self._accounts))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load paired accounts: %s", e)
            self._accounts = {}

    def _save(self) -> bool:
        """Atomically save paired accounts to disk."""
        try:
            # Ensure parent directory exists
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "accounts": [acc.to_dict() for acc in self._accounts.values()],
            }

            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=self._storage_path.parent,
            )
            try:
                import os
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)

                # Atomic rename
                Path(temp_path).replace(self._storage_path)
                logger.debug("Saved %d paired accounts", len(self._accounts))
                return True
            except Exception:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass
                raise

        except OSError as e:
            logger.error("Failed to save paired accounts: %s", e)
            return False

    @property
    def account_count(self) -> int:
        """Get number of paired accounts."""
        return len(self._accounts)

    def get_all_accounts(self) -> list[PairedAccount]:
        """Get all paired accounts."""
        return list(self._accounts.values())

    def get_account(self, user_id: str) -> Optional[PairedAccount]:
        """Get a specific account by user_id."""
        return self._accounts.get(user_id)

    def has_account(self, user_id: str) -> bool:
        """Check if an account is already paired."""
        return user_id in self._accounts

    def add_or_update_account(
        self,
        user_id: str,
        email: Optional[str] = None,
        id_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> PairedAccount:
        """Add a new account or update existing one.

        Args:
            user_id: Firebase user ID
            email: User email
            id_token: Firebase ID token
            refresh_token: Firebase refresh token

        Returns:
            The added/updated PairedAccount
        """
        now = datetime.now(timezone.utc).isoformat()

        existing = self._accounts.get(user_id)
        if existing:
            # Update existing
            account = PairedAccount(
                user_id=user_id,
                email=email or existing.email,
                id_token=id_token or existing.id_token,
                refresh_token=refresh_token or existing.refresh_token,
                paired_at=existing.paired_at,
                last_updated=now,
            )
        else:
            # Add new
            account = PairedAccount(
                user_id=user_id,
                email=email,
                id_token=id_token,
                refresh_token=refresh_token,
                paired_at=now,
                last_updated=now,
            )

        self._accounts[user_id] = account
        self._save()

        logger.info(
            "Paired account %s: %s",
            "updated" if existing else "added",
            email or user_id,
        )
        return account

    def remove_account(self, user_id: str) -> bool:
        """Remove a paired account.

        Args:
            user_id: Firebase user ID to remove

        Returns:
            True if account was removed, False if not found
        """
        if user_id not in self._accounts:
            return False

        del self._accounts[user_id]
        self._save()
        logger.info("Removed paired account: %s", user_id)
        return True

    async def refresh_account_token(
        self,
        user_id: str,
        api_key: str,
    ) -> Optional[str]:
        """Refresh the ID token for a specific account.

        Args:
            user_id: Firebase user ID
            api_key: Firebase Web API key

        Returns:
            New ID token if successful, None otherwise
        """
        account = self._accounts.get(user_id)
        if not account or not account.refresh_token:
            logger.warning("Cannot refresh token: account %s not found or no refresh token", user_id)
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://securetoken.googleapis.com/v1/token?key={api_key}",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": account.refresh_token,
                    },
                )

                if response.status_code != 200:
                    logger.error("Token refresh failed for %s: %s", user_id, response.text)
                    return None

                data = response.json()
                new_id_token = data.get("id_token")
                new_refresh_token = data.get("refresh_token")

                if new_id_token:
                    self._accounts[user_id] = account.with_updated_token(
                        id_token=new_id_token,
                        refresh_token=new_refresh_token,
                    )
                    self._save()
                    logger.debug("Refreshed token for account: %s", user_id)
                    return new_id_token

                return None

        except httpx.HTTPError as e:
            logger.error("Token refresh error for %s: %s", user_id, e)
            return None

    async def ensure_valid_token(
        self,
        user_id: str,
        api_key: str,
    ) -> tuple[Optional[str], bool]:
        """Return a usable ID token for an account, refreshing it if stale.

        Firebase ID tokens live for one hour. Before this existed, callers
        reused ``account.id_token`` verbatim however old it was, so every
        paired account started failing an hour after it was paired. This
        mirrors ``FirebaseAuthService.ensure_valid_token`` in service.py,
        which is why that class's writes kept working while these did not.

        A token is refreshed when it is absent, when its expiry cannot be
        read (a malformed or non-JWT value), or when it expires within
        ``TOKEN_REFRESH_THRESHOLD_SECONDS``.

        Returns:
            ``(id_token, refreshed)``. ``id_token`` is None when no usable
            token could be obtained; ``refreshed`` says whether the token
            was just re-issued, so callers know a retry would be pointless.
        """
        account = self._accounts.get(user_id)
        if account is None:
            return None, False

        id_token = account.id_token
        if not id_token:
            return await self.refresh_account_token(user_id, api_key), True

        expires_at = self._tokens.extract_token_expiration(id_token)
        if expires_at is None:
            # Not a decodable JWT. Never trust it — a stored value like this
            # is what makes Firestore answer 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.
            logger.info("Stored token for %s is not a readable JWT; refreshing", user_id)
            return await self.refresh_account_token(user_id, api_key), True

        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        if remaining < TOKEN_REFRESH_THRESHOLD_SECONDS:
            logger.info(
                "Token for %s expires in %.0fs, refreshing before write",
                user_id,
                remaining,
            )
            return await self.refresh_account_token(user_id, api_key), True

        return id_token, False

    async def update_url_for_all_accounts(
        self,
        project_id: str,
        server_id: str,
        api_key: str,
        tunnel_url: Optional[str] = None,
        local_url: Optional[str] = None,
    ) -> dict[str, bool]:
        """Update server URL in Firestore for all paired accounts.

        This should be called when:
        - Tunnel URL changes
        - Local IP changes

        Args:
            project_id: Firebase project ID
            server_id: Server identifier
            api_key: Firebase Web API key (for token refresh)
            tunnel_url: New tunnel URL (or None if not available)
            local_url: New local URL

        Returns:
            Dict of {user_id: success} for each account
        """
        results: dict[str, bool] = {}

        # Snapshot the items: refreshing a token replaces the stored account
        # object, and we do not want to mutate what we are iterating over.
        for user_id, account in list(self._accounts.items()):
            id_token, refreshed = await self.ensure_valid_token(user_id, api_key)

            if not id_token:
                logger.warning("No valid token for account %s, skipping URL update", user_id)
                results[user_id] = False
                continue

            # Update Firestore
            success = await device_registration.register_device(
                project_id=project_id,
                user_id=user_id,
                server_id=server_id,
                id_token=id_token,
                tunnel_url=tunnel_url,
                local_url=local_url,
            )

            if not success and not refreshed:
                # The token looked valid locally but Firestore rejected it —
                # clock skew, a revoked session, or a signing key that rotated
                # out. Refresh once and retry; register_device is an idempotent
                # PATCH with an updateMask, so replaying it is safe.
                logger.info(
                    "URL update rejected for %s with a locally-valid token; "
                    "refreshing once and retrying",
                    user_id,
                )
                retry_token = await self.refresh_account_token(user_id, api_key)
                if retry_token:
                    success = await device_registration.register_device(
                        project_id=project_id,
                        user_id=user_id,
                        server_id=server_id,
                        id_token=retry_token,
                        tunnel_url=tunnel_url,
                        local_url=local_url,
                    )

            results[user_id] = success
            if success:
                logger.info("Updated URL for account: %s", account.email or user_id)
            else:
                logger.warning("Failed to update URL for account: %s", account.email or user_id)

        return results


# Singleton instance
_paired_accounts_manager: Optional[PairedAccountsManager] = None


def get_paired_accounts_manager(
    firebase_config: Optional[dict[str, Any]] = None,
) -> PairedAccountsManager:
    """Get the singleton paired accounts manager instance."""
    global _paired_accounts_manager
    if _paired_accounts_manager is None:
        _paired_accounts_manager = PairedAccountsManager(firebase_config=firebase_config)
    return _paired_accounts_manager


def reset_paired_accounts_manager() -> None:
    """Reset the singleton instance (for testing)."""
    global _paired_accounts_manager
    _paired_accounts_manager = None
