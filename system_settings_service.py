"""Wrappers for system settings actions used by API routes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from database import get_settings_db
from heartbeat_settings import get_heartbeat_interval, set_heartbeat_interval
from llm_settings import (
    CODEX_SANDBOX_MODES,
    get_codex_sandbox_mode,
    get_llm_options_snapshot,
    set_codex_sandbox_mode,
    set_selected_llm,
)

# Setting keys
SETTING_ALLOW_IP_LOGIN = "allow_ip_login"

HEARTBEAT_MINUTES_MIN = 5
HEARTBEAT_MINUTES_MAX = 15


@dataclass(frozen=True)
class SystemSettingsResult:
    """Typed result for system-settings route responses."""

    success: bool
    status_code: int
    payload: dict[str, Any]

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize for route response helpers."""
        return self.payload


def get_heartbeat_settings_for_current_server() -> SystemSettingsResult:
    """Get heartbeat settings snapshot."""
    return SystemSettingsResult(
        success=True,
        status_code=200,
        payload={
            "interval_minutes": get_heartbeat_interval(),
            "min": HEARTBEAT_MINUTES_MIN,
            "max": HEARTBEAT_MINUTES_MAX,
        },
    )


def update_heartbeat_settings_for_current_server(interval_minutes: int) -> SystemSettingsResult:
    """Update heartbeat interval and return updated snapshot."""
    new_interval = set_heartbeat_interval(interval_minutes)
    return SystemSettingsResult(
        success=True,
        status_code=200,
        payload={
            "interval_minutes": new_interval,
            "min": HEARTBEAT_MINUTES_MIN,
            "max": HEARTBEAT_MINUTES_MAX,
        },
    )


def get_llm_options_for_current_server() -> SystemSettingsResult:
    """Return current LLM provider/model options."""
    return SystemSettingsResult(
        success=True,
        status_code=200,
        payload=get_llm_options_snapshot(),
    )


def update_llm_selection_for_current_server(company_id: str, model: str) -> SystemSettingsResult:
    """Update selected LLM provider/model with validation handling."""
    try:
        payload = set_selected_llm(company_id, model)
    except ValueError as exc:
        return SystemSettingsResult(
            success=False,
            status_code=400,
            payload={"error": str(exc)},
        )

    return SystemSettingsResult(success=True, status_code=200, payload=payload)


def get_codex_settings_for_current_server() -> SystemSettingsResult:
    """Get Codex-specific settings."""
    return SystemSettingsResult(
        success=True,
        status_code=200,
        payload={
            "sandbox_mode": get_codex_sandbox_mode(),
            "sandbox_modes": CODEX_SANDBOX_MODES,
        },
    )


def update_codex_settings_for_current_server(sandbox_mode: str) -> SystemSettingsResult:
    """Update Codex sandbox setting with validation handling."""
    try:
        payload = set_codex_sandbox_mode(sandbox_mode)
    except ValueError as exc:
        return SystemSettingsResult(
            success=False,
            status_code=400,
            payload={"error": str(exc)},
        )

    return SystemSettingsResult(success=True, status_code=200, payload=payload)


# ============================================================================
# IP Login Settings
# ============================================================================


def get_allow_ip_login() -> bool:
    """Get current IP login setting. Default is False (secure)."""
    settings_db = get_settings_db()
    value = settings_db.get(SETTING_ALLOW_IP_LOGIN, "false")
    return value.lower() == "true"


def set_allow_ip_login(enabled: bool) -> bool:
    """Set IP login setting. Returns new value."""
    settings_db = get_settings_db()
    settings_db.set(SETTING_ALLOW_IP_LOGIN, "true" if enabled else "false")
    return enabled


def get_ip_login_settings_for_current_server() -> SystemSettingsResult:
    """Get IP login settings snapshot."""
    return SystemSettingsResult(
        success=True,
        status_code=200,
        payload={
            "allow_ip_login": get_allow_ip_login(),
            "warning": "When enabled, anyone on your network can access without QR pairing. Use only for development/testing.",
        },
    )


async def update_ip_login_settings_for_current_server(
    allow_ip_login: bool,
    *,
    stop_tunnel_func: Optional[Any] = None,
) -> SystemSettingsResult:
    """Update IP login setting and return updated snapshot.

    When IP Login is enabled, automatically stops the tunnel for security.
    External access via tunnel should not be allowed in IP Login mode.
    """
    new_value = set_allow_ip_login(allow_ip_login)

    tunnel_stopped = False
    if new_value:
        # IP Login ON → Stop tunnel for security
        # Avoid circular import by lazy importing
        if stop_tunnel_func is None:
            from optional_services import get_tunnel_service

            tunnel_service = get_tunnel_service()
            if tunnel_service and tunnel_service.is_running:
                await tunnel_service.stop()
                tunnel_stopped = True
        else:
            # For testing
            result = await stop_tunnel_func()
            tunnel_stopped = result

    message = (
        "IP login enabled. Anyone on your network can now access without pairing."
        if new_value
        else "IP login disabled. QR pairing is required for access."
    )
    if tunnel_stopped:
        message += " Tunnel has been stopped for security."

    return SystemSettingsResult(
        success=True,
        status_code=200,
        payload={
            "allow_ip_login": new_value,
            "tunnel_stopped": tunnel_stopped,
            "message": message,
        },
    )
