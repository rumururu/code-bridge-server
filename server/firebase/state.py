"""Immutable state management for Firebase authentication.

This module provides thread-safe, atomic state persistence that ensures
server_id is never lost when credentials change.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuthState(Enum):
    """Firebase authentication state machine.

    State transitions:
    - UNINITIALIZED -> LOADING (on initialize() call)
    - LOADING -> LOADED (after state loaded from disk)
    - LOADING -> ERROR (on load failure)
    - LOADED -> AUTHENTICATED (after successful token verification)
    - AUTHENTICATED -> LOADED (on token expiration/invalidation)
    - AUTHENTICATED -> ERROR (on unrecoverable error)
    - Any state -> UNINITIALIZED (on shutdown/reset)
    """
    UNINITIALIZED = auto()  # Service created, not started
    LOADING = auto()        # Loading config/state from disk
    LOADED = auto()         # State loaded, but not authenticated
    AUTHENTICATED = auto()  # Token verified, user authenticated
    ERROR = auto()          # Unrecoverable error state


@dataclass(frozen=True)
class PersistedState:
    """Immutable state persisted to server_info.json.

    This dataclass is frozen (immutable) to prevent accidental modifications.
    All state changes create new instances, ensuring atomic updates.

    The server_id is the stable identifier that MUST be preserved across
    all state changes. Credential changes create new instances that
    preserve the original server_id.
    """
    server_id: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    id_token: Optional[str] = None
    refresh_token: Optional[str] = None

    def with_credentials(
        self,
        *,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        id_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> PersistedState:
        """Create new state with updated credentials, preserving server_id.

        This is the ONLY way to update credentials. The server_id is always
        preserved from the current instance.

        Args:
            user_id: Firebase user ID
            email: User email
            id_token: Firebase ID token
            refresh_token: Firebase refresh token

        Returns:
            New PersistedState with updated fields, same server_id
        """
        return PersistedState(
            server_id=self.server_id,  # Always preserve!
            user_id=user_id if user_id is not None else self.user_id,
            email=email if email is not None else self.email,
            id_token=id_token if id_token is not None else self.id_token,
            refresh_token=refresh_token if refresh_token is not None else self.refresh_token,
        )

    def without_credentials(self) -> PersistedState:
        """Create new state with credentials cleared, preserving server_id.

        Use this for complete logout/disconnect. The server_id is preserved
        so the server maintains its identity for future re-authentication.

        Returns:
            New PersistedState with only server_id
        """
        return PersistedState(server_id=self.server_id)

    def without_id_token(self) -> PersistedState:
        """Create new state with ID token cleared but other fields preserved.

        Use this when ID token expires but refresh_token and user ownership
        info should be kept for retry capability.

        Returns:
            New PersistedState with id_token cleared
        """
        return PersistedState(
            server_id=self.server_id,
            user_id=self.user_id,
            email=self.email,
            id_token=None,  # Clear ID token
            refresh_token=self.refresh_token,  # Keep for retry
        )

    @property
    def has_credentials(self) -> bool:
        """Check if state has valid credentials for authentication."""
        return self.user_id is not None and self.id_token is not None

    @property
    def has_refresh_capability(self) -> bool:
        """Check if state can refresh tokens."""
        return self.refresh_token is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Only includes non-None fields to keep the file clean.
        """
        data: dict[str, Any] = {"server_id": self.server_id}
        if self.user_id:
            data["user_id"] = self.user_id
        if self.email:
            data["email"] = self.email
        if self.id_token:
            data["id_token"] = self.id_token
        if self.refresh_token:
            data["refresh_token"] = self.refresh_token
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional[PersistedState]:
        """Create PersistedState from dictionary.

        Args:
            data: Dictionary with state fields

        Returns:
            PersistedState if server_id exists, None otherwise
        """
        server_id = data.get("server_id")
        if not server_id:
            return None

        return cls(
            server_id=server_id,
            user_id=data.get("user_id"),
            email=data.get("email"),
            id_token=data.get("id_token"),
            refresh_token=data.get("refresh_token"),
        )

    def save(self, path: Path) -> bool:
        """Atomically save state to file.

        Uses write-to-temp-then-rename pattern to prevent partial writes.

        Args:
            path: Path to save state file

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=path.parent,
            )
            try:
                import os
                with os.fdopen(fd, "w") as f:
                    json.dump(self.to_dict(), f, indent=2)

                # Atomic rename
                Path(temp_path).replace(path)
                logger.debug("State saved atomically to %s", path)
                return True
            except Exception:
                # Clean up temp file on error
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass
                raise

        except OSError as e:
            logger.error("Failed to save state to %s: %s", path, e)
            return False

    @classmethod
    def load(cls, path: Path) -> Optional[PersistedState]:
        """Load state from file.

        Args:
            path: Path to state file

        Returns:
            PersistedState if file exists and is valid, None otherwise
        """
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load state from %s: %s", path, e)
            return None


@dataclass(frozen=True)
class LegacyDeviceInfo:
    """Legacy device_info.json format for migration."""
    device_id: str  # Maps to server_id
    user_id: Optional[str] = None
    email: Optional[str] = None
    id_token: Optional[str] = None
    refresh_token: Optional[str] = None

    def to_persisted_state(self) -> PersistedState:
        """Convert legacy format to new PersistedState."""
        return PersistedState(
            server_id=self.device_id,  # device_id -> server_id
            user_id=self.user_id,
            email=self.email,
            id_token=self.id_token,
            refresh_token=self.refresh_token,
        )

    @classmethod
    def load(cls, path: Path) -> Optional[LegacyDeviceInfo]:
        """Load legacy device info from file."""
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)

            device_id = data.get("device_id")
            if not device_id:
                return None

            return cls(
                device_id=device_id,
                user_id=data.get("user_id"),
                email=data.get("email"),
                id_token=data.get("id_token"),
                refresh_token=data.get("refresh_token"),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load legacy device info from %s: %s", path, e)
            return None
