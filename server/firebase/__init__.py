"""Firebase authentication package for Code Bridge Server.

This package provides Firebase token verification and Firestore operations
with explicit state management to prevent save-before-load bugs.

Public API:
-----------
- get_firebase_auth() -> FirebaseAuthService
    Get or create the singleton Firebase auth service.

- is_firebase_available() -> bool
    Check if Firebase is configured and available.

- get_server_id() -> Optional[str]
    Safely get server ID without triggering side effects.
    Returns None if service is not initialized.

Usage:
------
    from firebase import get_firebase_auth, is_firebase_available

    if is_firebase_available():
        auth = get_firebase_auth()
        await auth.initialize()  # MUST call before using!

        if auth.is_authenticated:
            print(f"Server ID: {auth.server_id}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .service import (
    FirebaseAuthService,
    FirebaseAuthServiceError,
    NotInitializedError,
    FIREBASE_CONFIG_PATH,
    SERVER_INFO_PATH,
)
from .state import AuthState, PersistedState
from .paired_accounts import (
    PairedAccount,
    PairedAccountsManager,
    get_paired_accounts_manager,
    reset_paired_accounts_manager,
)

__all__ = [
    # Main service
    "FirebaseAuthService",
    "get_firebase_auth",
    # Helper functions
    "is_firebase_available",
    "get_server_id",
    # State types
    "AuthState",
    "PersistedState",
    # Exceptions
    "FirebaseAuthServiceError",
    "NotInitializedError",
    # Multi-account support
    "PairedAccount",
    "PairedAccountsManager",
    "get_paired_accounts_manager",
    "reset_paired_accounts_manager",
]

logger = logging.getLogger(__name__)

# Singleton instance
_firebase_auth: Optional[FirebaseAuthService] = None


def get_firebase_auth() -> FirebaseAuthService:
    """Get or create Firebase auth service singleton.

    Note: You MUST call initialize() on the returned service before
    using any methods or accessing authenticated properties.

    Returns:
        FirebaseAuthService singleton instance
    """
    global _firebase_auth
    if _firebase_auth is None:
        _firebase_auth = FirebaseAuthService()
    return _firebase_auth


def is_firebase_available() -> bool:
    """Check if Firebase is configured and available.

    This checks for the existence of firebase_config.json without
    creating the service or loading any state.

    Returns:
        True if Firebase config exists, False otherwise
    """
    return FIREBASE_CONFIG_PATH.exists()


def get_server_id() -> Optional[str]:
    """Safely get server ID without triggering side effects.

    This method safely retrieves the server_id by:
    1. If service is initialized: Return from service
    2. If service not initialized: Load directly from state file

    This avoids the save-before-load bug where accessing server_id
    could trigger state saving that overwrites user_id.

    Returns:
        Server ID string, or None if not available
    """
    global _firebase_auth

    # If service exists and is initialized, use it
    if _firebase_auth is not None and _firebase_auth.is_initialized:
        try:
            return _firebase_auth.server_id
        except (FirebaseAuthServiceError, NotInitializedError):
            pass

    # Otherwise, load directly from state file
    state = PersistedState.load(SERVER_INFO_PATH)
    if state:
        return state.server_id

    return None


def reset_singleton() -> None:
    """Reset the singleton instance (for testing only).

    WARNING: This should only be used in tests. Do not use in production.
    """
    global _firebase_auth
    _firebase_auth = None
