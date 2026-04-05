"""Backward compatibility stub - import from pairing_service instead."""
from .pairing_models import (
    CurrentPairingDataResult,
    FirebaseUserInfo,
    PairingClientStatus,
    PairingCodeVerifyResult,
    PairingData,
    PairingOperationResult,
    PairingPageContextResult,
    PairingQrResult,
    PairingRevokeResult,
    PairingStatus,
    PairingVerifyTokenResult,
    PairTokenStatus,
)
from .pairing_service import (
    PAIRING_TOKEN_TTL_SECONDS,
    RATE_LIMIT_LOCKOUT_SECONDS,
    PairingService,
    SSOPairingResult,
    build_current_pairing_data_result,
    build_current_pairing_page_context_result,
    build_current_pairing_qr_result,
    create_current_pairing_data,
    get_pair_token_status_for_current_server,
    get_pairing_service,
    get_pairing_status_for_current_server,
    revoke_paired_client_for_current_server,
    verify_pairing_code_for_current_server,
    verify_sso_pairing_for_current_server,
)
from core.rate_limiter import (
    RATE_LIMIT_BLOCK_SECONDS,
    RATE_LIMIT_MAX_ATTEMPTS,
    RATE_LIMIT_WINDOW_SECONDS,
    RateLimitEntry,
    RateLimiter,
)

__all__ = [
    # Constants
    "PAIRING_TOKEN_TTL_SECONDS",
    "RATE_LIMIT_LOCKOUT_SECONDS",
    "RATE_LIMIT_BLOCK_SECONDS",
    "RATE_LIMIT_MAX_ATTEMPTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    # Models from pairing_models
    "CurrentPairingDataResult",
    "FirebaseUserInfo",
    "PairingClientStatus",
    "PairingCodeVerifyResult",
    "PairingData",
    "PairingOperationResult",
    "PairingPageContextResult",
    "PairingQrResult",
    "PairingRevokeResult",
    "PairingStatus",
    "PairingVerifyTokenResult",
    "PairTokenStatus",
    # Classes from pairing_service
    "PairingService",
    "SSOPairingResult",
    # Functions from pairing_service
    "build_current_pairing_data_result",
    "build_current_pairing_page_context_result",
    "build_current_pairing_qr_result",
    "create_current_pairing_data",
    "get_pair_token_status_for_current_server",
    "get_pairing_service",
    "get_pairing_status_for_current_server",
    "revoke_paired_client_for_current_server",
    "verify_pairing_code_for_current_server",
    "verify_sso_pairing_for_current_server",
    # Rate limiter classes
    "RateLimitEntry",
    "RateLimiter",
]
