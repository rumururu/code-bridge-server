"""Pairing service package.

Provides QR code pairing, token verification, and API key management.
"""

from pairing.pairing_service import (
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

__all__ = [
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
]
