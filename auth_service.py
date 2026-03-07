"""Common API key authentication helpers for routes and dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import get_config
from pairing import get_pairing_service


@dataclass(frozen=True)
class ApiKeyValidationResult:
    """Result of API key validation."""

    success: bool
    api_key: str | None = None
    error: str | None = None
    is_ip_login: bool = False  # True when access granted via IP login (no api_key)


def _check_allow_ip_login() -> bool:
    """Check if IP login is enabled. Isolated to avoid circular imports."""
    from system_settings_service import get_allow_ip_login

    return get_allow_ip_login()


def validate_api_key_for_current_server(
    api_key: str | None,
    *,
    pairing_service: Any | None = None,
    config: Any | None = None,
) -> ApiKeyValidationResult:
    """Validate API key against pairing service.

    Access is granted in these cases (checked in order):
    1. IP Login is enabled -> allow without api_key (local only, tunnel blocked in deps.py)
    2. Static api_key from config.yaml matches -> allow
    3. Paired client api_key validates -> allow
    """
    resolved_config = config or get_config()
    configured_api_key = str(getattr(resolved_config, "api_key", "") or "").strip()

    # Check if IP login is enabled (allows anonymous LOCAL access)
    if _check_allow_ip_login():
        return ApiKeyValidationResult(
            success=True,
            api_key=api_key or "__ip_login__",
            is_ip_login=api_key is None,
        )

    # Check static API key from config (for advanced users)
    if configured_api_key and api_key == configured_api_key:
        return ApiKeyValidationResult(success=True, api_key=api_key)

    # No API key provided
    if not api_key:
        return ApiKeyValidationResult(success=False, error="API key required")

    # Check paired client API keys
    resolved_pairing = pairing_service or get_pairing_service()
    if resolved_pairing.validate_api_key(api_key):
        return ApiKeyValidationResult(success=True, api_key=api_key)

    return ApiKeyValidationResult(success=False, error="Invalid API key")
