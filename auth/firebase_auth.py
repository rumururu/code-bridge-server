"""Firebase Token Verification for Code Bridge Server.

DEPRECATED: This module is maintained for backward compatibility.
New code should import from the `firebase` package directly:

    from firebase import get_firebase_auth, is_firebase_available

This module re-exports from the firebase package and adds deprecation warnings.
"""

from __future__ import annotations

import warnings
from pathlib import Path

# Re-export everything from the new firebase package
from firebase import (
    FirebaseAuthService,
    get_firebase_auth,
    is_firebase_available,
    get_server_id,
    AuthState,
    PersistedState,
    FirebaseAuthServiceError,
    NotInitializedError,
)
from firebase.service import (
    FIREBASE_CONFIG_PATH,
    SERVER_INFO_PATH,
    LEGACY_DEVICE_INFO_PATH,
)

# Re-export for backward compatibility
__all__ = [
    "FirebaseAuthService",
    "get_firebase_auth",
    "is_firebase_available",
    "get_server_id",
    "FIREBASE_CONFIG_PATH",
    "SERVER_INFO_PATH",
    "LEGACY_DEVICE_INFO_PATH",
    # Legacy names
    "GOOGLE_CERTS_URL",
]

# Keep this for any code that imports it directly
GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"


def _emit_deprecation_warning():
    """Emit deprecation warning for direct imports from firebase_auth."""
    warnings.warn(
        "Importing from 'firebase_auth' is deprecated. "
        "Use 'from firebase import get_firebase_auth' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# The actual singleton and function are in firebase/__init__.py
# This file just re-exports them for backward compatibility.
