"""System settings routes: heartbeat, LLM configuration, Firebase auth, and IP login."""

from typing import Any

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import JSONResponse, Response
from auth.firebase_auth import get_firebase_auth
from llm.llm_commands import execute_llm_command, get_llm_command_snapshot
from models import (
    CodexSettingsUpdate,
    IpLoginUpdate,
    LlmAccessUpdate,
    LlmProviderInstallRequest,
    LlmSelectionUpdate,
)
from system.system_settings_service import (
    cancel_llm_provider_install_job_for_current_server,
    get_codex_settings_for_current_server,
    get_heartbeat_settings_for_current_server,
    get_ip_login_settings_for_current_server,
    get_llm_provider_install_job_for_current_server,
    get_llm_options_for_current_server,
    install_llm_provider_for_current_server,
    update_codex_settings_for_current_server,
    update_heartbeat_settings_for_current_server,
    update_ip_login_settings_for_current_server,
    update_llm_access_for_current_server,
    update_llm_selection_for_current_server,
)
from .deps import verify_api_key, verify_api_key_or_localhost
from .result_response import as_route_response

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/heartbeat", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_heartbeat_settings() -> dict[str, Any] | Response:
    """Get current heartbeat settings."""
    result = get_heartbeat_settings_for_current_server()
    return as_route_response(result)


@router.put("/heartbeat", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_heartbeat_settings(interval_minutes: int) -> dict[str, Any] | Response:
    """Update heartbeat interval (5-15 minutes)."""
    result = update_heartbeat_settings_for_current_server(interval_minutes)
    return as_route_response(result)


@router.get("/llm/options", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_llm_options() -> dict[str, Any] | Response:
    """Return available LLM providers and models with current selection."""
    result = get_llm_options_for_current_server()
    return as_route_response(result)


@router.get("/llm/commands", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_llm_commands(
    provider_id: str | None = None,
    model: str | None = None,
    scope: str = Query("project", pattern="^(global|project)$"),
    refresh: bool = False,
) -> dict[str, Any]:
    """Return Code Bridge and discovered provider slash commands."""
    return get_llm_command_snapshot(
        provider_id=provider_id,
        model=model,
        scope=scope,
        refresh=refresh,
    )


@router.post("/llm/commands/execute", dependencies=[Depends(verify_api_key)], response_model=None)
async def execute_llm_command_route(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Execute a Code Bridge slash command."""
    return execute_llm_command(
        name=str(payload.get("name") or payload.get("command") or ""),
        provider_id=payload.get("provider_id"),
        model=payload.get("model"),
        scope=str(payload.get("scope") or "project"),
        project_name=payload.get("project_name"),
    )


@router.put("/llm/selection", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_llm_selection(payload: LlmSelectionUpdate) -> dict[str, Any] | Response:
    """Select active LLM provider and model for chat."""
    result = update_llm_selection_for_current_server(payload.company_id, payload.model)
    return as_route_response(result)


# The dashboard is the place this is set, and it talks from localhost with no
# API key — the same reason ip-login uses this dependency.
@router.put(
    "/llm/access",
    dependencies=[Depends(verify_api_key_or_localhost)],
    response_model=None,
)
async def update_llm_access(payload: LlmAccessUpdate) -> dict[str, Any] | Response:
    """Switch one LLM provider on or off for every model picker."""
    result = update_llm_access_for_current_server(payload.company_id, payload.enabled)
    return as_route_response(result)


@router.post("/llm/providers/{provider_id}/install", dependencies=[Depends(verify_api_key)], response_model=None)
async def install_llm_provider(provider_id: str, payload: LlmProviderInstallRequest) -> dict[str, Any] | Response:
    """Start an async job to install one supported LLM provider CLI."""
    result = await install_llm_provider_for_current_server(provider_id, payload.method)
    if result.success:
        return JSONResponse(status_code=result.status_code, content=result.payload)
    return as_route_response(result)


@router.get("/llm/providers/install/jobs/{job_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_llm_provider_install_job(job_id: str) -> dict[str, Any] | Response:
    """Return provider CLI install job status."""
    result = get_llm_provider_install_job_for_current_server(job_id)
    return as_route_response(result)


@router.post("/llm/providers/install/jobs/{job_id}/cancel", dependencies=[Depends(verify_api_key)], response_model=None)
async def cancel_llm_provider_install_job(job_id: str) -> dict[str, Any] | Response:
    """Cancel a queued/running provider CLI install job."""
    result = await cancel_llm_provider_install_job_for_current_server(job_id)
    return as_route_response(result)


@router.get("/llm/codex/settings", dependencies=[Depends(verify_api_key)], response_model=None)
@router.get("/llm/agent/settings", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_codex_settings() -> dict[str, Any] | Response:
    """Get Codex-specific settings."""
    result = get_codex_settings_for_current_server()
    return as_route_response(result)


@router.put("/llm/codex/settings", dependencies=[Depends(verify_api_key)], response_model=None)
@router.put("/llm/agent/settings", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_codex_settings(payload: CodexSettingsUpdate) -> dict[str, Any] | Response:
    """Update Codex-specific settings."""
    result = update_codex_settings_for_current_server(payload.sandbox_mode)
    return as_route_response(result)


@router.post("/firebase/logout", dependencies=[Depends(verify_api_key)])
async def firebase_logout() -> dict[str, Any]:
    """Logout from Firebase and clear authentication data."""
    firebase_auth = get_firebase_auth()
    success = await firebase_auth.clear_auth()
    if success:
        return {"success": True, "message": "Logged out from Firebase"}
    return {"success": False, "error": "Logout failed"}


@router.get("/ip-login", dependencies=[Depends(verify_api_key_or_localhost)], response_model=None)
async def get_ip_login_settings() -> dict[str, Any] | Response:
    """Get current IP login settings."""
    result = get_ip_login_settings_for_current_server()
    return as_route_response(result)


@router.put("/ip-login", dependencies=[Depends(verify_api_key_or_localhost)], response_model=None)
async def update_ip_login_settings(payload: IpLoginUpdate) -> dict[str, Any] | Response:
    """Update IP login setting.

    WARNING: When enabled, anyone on your network can access without QR pairing.
    When IP Login is enabled, tunnel is automatically stopped for security.
    Use only for development/testing.
    """
    result = await update_ip_login_settings_for_current_server(payload.allow_ip_login)
    return as_route_response(result)
