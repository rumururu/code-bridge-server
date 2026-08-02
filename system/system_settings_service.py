"""Wrappers for system settings actions used by API routes."""

from __future__ import annotations

from typing import Any, Optional

from core.database import get_settings_db
from remote.heartbeat_settings import get_heartbeat_interval, set_heartbeat_interval
from llm.llm_settings import (
    CODEX_SANDBOX_MODES,
    get_codex_sandbox_mode,
    get_llm_options_snapshot,
    set_codex_sandbox_mode,
    set_company_enabled,
    set_selected_llm,
)
from core.base_result import BaseRouteResult
from system.llm_provider_install_jobs import (
    INVALID_INSTALL_METHOD,
    UNKNOWN_PROVIDER,
    get_llm_provider_install_job_manager,
)

# Setting keys
SETTING_ALLOW_IP_LOGIN = "allow_ip_login"

HEARTBEAT_MINUTES_MIN = 5
HEARTBEAT_MINUTES_MAX = 15

# Backwards-compatible alias
SystemSettingsResult = BaseRouteResult


def get_heartbeat_settings_for_current_server() -> SystemSettingsResult:
    """Get heartbeat settings snapshot."""
    return SystemSettingsResult.ok({
        "interval_minutes": get_heartbeat_interval(),
        "min": HEARTBEAT_MINUTES_MIN,
        "max": HEARTBEAT_MINUTES_MAX,
    })


def update_heartbeat_settings_for_current_server(interval_minutes: int) -> SystemSettingsResult:
    """Update heartbeat interval and return updated snapshot."""
    new_interval = set_heartbeat_interval(interval_minutes)
    return SystemSettingsResult.ok({
        "interval_minutes": new_interval,
        "min": HEARTBEAT_MINUTES_MIN,
        "max": HEARTBEAT_MINUTES_MAX,
    })


def get_llm_options_for_current_server() -> SystemSettingsResult:
    """Return current LLM provider/model options."""
    return SystemSettingsResult.ok(get_llm_options_snapshot())


def update_llm_selection_for_current_server(company_id: str, model: str) -> SystemSettingsResult:
    """Update selected LLM provider/model with validation handling."""
    try:
        payload = set_selected_llm(company_id, model)
    except ValueError as exc:
        return SystemSettingsResult.error(400, str(exc))

    return SystemSettingsResult.ok(payload)


def update_llm_access_for_current_server(
    company_id: str, enabled: bool
) -> SystemSettingsResult:
    """Switch a provider on or off, refusing to leave nothing usable."""
    try:
        payload = set_company_enabled(company_id, enabled)
    except ValueError as exc:
        return SystemSettingsResult.error(400, str(exc))

    return SystemSettingsResult.ok(payload)


def _parse_install_validation_error(message: str) -> tuple[str | None, str]:
    if ": " not in message:
        return None, message
    code, detail = message.split(": ", 1)
    if code in {UNKNOWN_PROVIDER, INVALID_INSTALL_METHOD}:
        return code, detail
    return None, message


async def install_llm_provider_for_current_server(provider_id: str, method: str) -> SystemSettingsResult:
    """Start an async job to install a supported LLM provider CLI."""
    manager = get_llm_provider_install_job_manager()
    try:
        job = await manager.start_install(provider_id, method)
    except ValueError as exc:
        error_code, detail = _parse_install_validation_error(str(exc))
        return SystemSettingsResult.error(400, detail, error_code=error_code)

    return SystemSettingsResult.ok(manager.serialize(job), status_code=202)


def get_llm_provider_install_job_for_current_server(job_id: str) -> SystemSettingsResult:
    """Return an async provider install job status."""
    manager = get_llm_provider_install_job_manager()
    job = manager.get_job(job_id)
    if job is None:
        return SystemSettingsResult.error(404, "Install job not found")
    return SystemSettingsResult.ok(manager.serialize(job))


async def cancel_llm_provider_install_job_for_current_server(job_id: str) -> SystemSettingsResult:
    """Cancel a queued/running provider install job."""
    manager = get_llm_provider_install_job_manager()
    job = await manager.cancel_job(job_id)
    if job is None:
        return SystemSettingsResult.error(404, "Install job not found")
    return SystemSettingsResult.ok(manager.serialize(job))


def get_codex_settings_for_current_server() -> SystemSettingsResult:
    """Get Codex-specific settings."""
    return SystemSettingsResult.ok({
        "sandbox_mode": get_codex_sandbox_mode(),
        "sandbox_modes": CODEX_SANDBOX_MODES,
    })


def update_codex_settings_for_current_server(sandbox_mode: str) -> SystemSettingsResult:
    """Update Codex sandbox setting with validation handling."""
    try:
        payload = set_codex_sandbox_mode(sandbox_mode)
    except ValueError as exc:
        return SystemSettingsResult.error(400, str(exc))

    return SystemSettingsResult.ok(payload)


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
    return SystemSettingsResult.ok({
        "allow_ip_login": get_allow_ip_login(),
        "warning": "When enabled, anyone on your network can access without QR pairing. Use only for development/testing.",
    })


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
            from system.optional_services import get_tunnel_service

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

    return SystemSettingsResult.ok({
        "allow_ip_login": new_value,
        "tunnel_stopped": tunnel_stopped,
        "message": message,
    })
