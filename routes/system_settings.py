"""System settings routes: heartbeat, LLM configuration, Firebase auth, and IP login."""

from fastapi import APIRouter, Depends
from firebase_auth import get_firebase_auth
from models import CodexSettingsUpdate, IpLoginUpdate, LlmSelectionUpdate
from system_settings_service import (
    get_codex_settings_for_current_server,
    get_heartbeat_settings_for_current_server,
    get_ip_login_settings_for_current_server,
    get_llm_options_for_current_server,
    update_codex_settings_for_current_server,
    update_heartbeat_settings_for_current_server,
    update_ip_login_settings_for_current_server,
    update_llm_selection_for_current_server,
)
from .deps import verify_api_key, verify_api_key_or_localhost
from .result_response import as_route_response

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/heartbeat", dependencies=[Depends(verify_api_key)])
async def get_heartbeat_settings():
    """Get current heartbeat settings."""
    result = get_heartbeat_settings_for_current_server()
    return as_route_response(result)


@router.put("/heartbeat", dependencies=[Depends(verify_api_key)])
async def update_heartbeat_settings(interval_minutes: int):
    """Update heartbeat interval (5-15 minutes)."""
    result = update_heartbeat_settings_for_current_server(interval_minutes)
    return as_route_response(result)


@router.get("/llm/options", dependencies=[Depends(verify_api_key)])
async def get_llm_options():
    """Return available LLM providers and models with current selection."""
    result = get_llm_options_for_current_server()
    return as_route_response(result)


@router.put("/llm/selection", dependencies=[Depends(verify_api_key)])
async def update_llm_selection(payload: LlmSelectionUpdate):
    """Select active LLM provider and model for chat."""
    result = update_llm_selection_for_current_server(payload.company_id, payload.model)
    return as_route_response(result)


@router.get("/llm/codex/settings", dependencies=[Depends(verify_api_key)])
async def get_codex_settings():
    """Get Codex-specific settings."""
    result = get_codex_settings_for_current_server()
    return as_route_response(result)


@router.put("/llm/codex/settings", dependencies=[Depends(verify_api_key)])
async def update_codex_settings(payload: CodexSettingsUpdate):
    """Update Codex-specific settings."""
    result = update_codex_settings_for_current_server(payload.sandbox_mode)
    return as_route_response(result)


@router.post("/firebase/logout", dependencies=[Depends(verify_api_key)])
async def firebase_logout():
    """Logout from Firebase and clear authentication data."""
    firebase_auth = get_firebase_auth()
    success = await firebase_auth.clear_auth()
    if success:
        return {"success": True, "message": "Logged out from Firebase"}
    return {"success": False, "error": "Logout failed"}


@router.get("/ip-login", dependencies=[Depends(verify_api_key_or_localhost)])
async def get_ip_login_settings():
    """Get current IP login settings."""
    result = get_ip_login_settings_for_current_server()
    return as_route_response(result)


@router.put("/ip-login", dependencies=[Depends(verify_api_key_or_localhost)])
async def update_ip_login_settings(payload: IpLoginUpdate):
    """Update IP login setting.

    WARNING: When enabled, anyone on your network can access without QR pairing.
    When IP Login is enabled, tunnel is automatically stopped for security.
    Use only for development/testing.
    """
    result = await update_ip_login_settings_for_current_server(payload.allow_ip_login)
    return as_route_response(result)
