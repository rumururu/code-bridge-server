"""Services package for Code Bridge server.

This package provides a unified access point for all service modules.
Import directly from this package for backward compatibility:

    from services import PairingService, get_pairing_service
    from services import ProjectManager, get_project_manager
    from services import FileManager, get_file_manager
    from services import PreviewProxy, get_preview_proxy
"""

# Re-export from root modules for backward compatibility
from pairing_service import (
    PairingService,
    PairingData,
    PairingStatus,
    PairingClientStatus,
    PairingOperationResult,
    PairingVerifyTokenResult,
    PairingQrResult,
    PairingCodeVerifyResult,
    PairingPageContextResult,
    PairingRevokeResult,
    CurrentPairingDataResult,
    SSOPairingResult,
    FirebaseUserInfo,
    RateLimiter,
    RateLimitEntry,
    get_pairing_service,
)

from project_manager import (
    ProjectManager,
    get_project_manager,
)

from file_manager import (
    FileManager,
    get_file_manager,
)

from preview_service import (
    PreviewProxy,
    get_preview_proxy,
)

__all__ = [
    # Pairing
    "PairingService",
    "PairingData",
    "PairingStatus",
    "PairingClientStatus",
    "PairingOperationResult",
    "PairingVerifyTokenResult",
    "PairingQrResult",
    "PairingCodeVerifyResult",
    "PairingPageContextResult",
    "PairingRevokeResult",
    "CurrentPairingDataResult",
    "SSOPairingResult",
    "FirebaseUserInfo",
    "RateLimiter",
    "RateLimitEntry",
    "get_pairing_service",
    # Project
    "ProjectManager",
    "get_project_manager",
    # File
    "FileManager",
    "get_file_manager",
    # Preview
    "PreviewProxy",
    "get_preview_proxy",
]
