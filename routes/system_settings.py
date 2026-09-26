"""System settings routes: heartbeat, LLM configuration, Firebase auth, and IP login."""

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from audit.route_audit import record_api_action
from auth.firebase_auth import get_firebase_auth
from llm.llm_commands import execute_llm_command, get_llm_command_snapshot
from models import (
    CodexSettingsUpdate,
    IpLoginUpdate,
    LlmAccessUpdate,
    LlmProviderInstallRequest,
    LlmSelectionUpdate,
    McpServerUpsert,
)
from system import mcp_registry
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


# Same dependency, and the same reason, as `/llm/access` below: the dashboard
# is one of the places this is set, and it talks from localhost with no API
# key. It used to be key-only, which left the agents page able to *report* that
# the selected provider was out of quota but unable to do anything about it —
# an offer to switch that could not switch. `verify_api_key_or_localhost`
# refuses tunnel traffic explicitly (routes/deps.py), so widening this does not
# widen the external surface.
@router.put(
    "/llm/selection",
    dependencies=[Depends(verify_api_key_or_localhost)],
    response_model=None,
)
async def update_llm_selection(payload: LlmSelectionUpdate) -> dict[str, Any] | Response:
    """Select active LLM provider and model for chat."""
    result = update_llm_selection_for_current_server(payload.company_id, payload.model)
    return as_route_response(result)


# The dashboard is the place this is set, and it talks from localhost with no
# API key — the same reason ip-login uses this dependency.
class BuilderMcpToggle(BaseModel):
    """Whether the Agent Builder conversation may ask to use an MCP tool."""

    enabled: bool


@router.get(
    "/builder/mcp-tools",
    dependencies=[Depends(verify_api_key_or_localhost)],
    response_model=None,
)
async def get_builder_mcp_tools() -> dict[str, Any]:
    """Report the setting, and what it would reach.

    The server names are part of the answer, not decoration: "allow the
    design conversation to use MCP tools" means nothing until the reader can
    see *which* servers this machine has. Same registry an agent run reads,
    so the two can never disagree.
    """

    from agent.builder_tool_policy import builder_mcp_enabled
    from agent.capability_registry import detected_mcp_server_configs

    return {
        "enabled": builder_mcp_enabled(),
        "servers": sorted(detected_mcp_server_configs()),
    }


@router.put(
    "/builder/mcp-tools",
    dependencies=[Depends(verify_api_key_or_localhost)],
    response_model=None,
)
async def update_builder_mcp_tools(payload: BuilderMcpToggle) -> dict[str, Any]:
    """Switch the asking on or off.

    Off is the default and the safe direction: with it off the servers are not
    even attached to the builder session, so a design conversation cannot see
    a tool, and a call it invents anyway is refused without troubling anyone.

    On does not grant anything by itself — each call is still put to the
    person, one at a time. Shell and file tools stay refused either way; that
    boundary is drawn by prefix in `builder_tool_policy` and is not a setting.
    """

    from agent.builder_tool_policy import set_builder_mcp_enabled
    from agent.capability_registry import detected_mcp_server_configs

    set_builder_mcp_enabled(payload.enabled)
    record_api_action(
        operation="provider.tool",
        details={
            "surface": "agent_builder_settings",
            "builder_mcp_enabled": payload.enabled,
        },
        success=True,
        status_code=200,
    )
    return {
        "enabled": payload.enabled,
        "servers": sorted(detected_mcp_server_configs()),
    }


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


# --- MCP servers registered with Code Bridge ---------------------------------
# Before these existed, attaching an MCP server to an agent meant installing the
# Claude Code CLI and hand-editing its `~/.claude.json`
# (`agent/capability_registry.py::_mcp_config_paths`), which a phone app cannot
# ask anyone to do. These write to Code Bridge's own store instead; detection,
# the tool picker, and `mcp_tool` step gating all pick them up from there.
#
# `verify_api_key`, not `verify_api_key_or_localhost`: the phone app is the
# caller, and unlike ip-login and llm/access there is no dashboard form behind
# this. A registration body carries credentials, so the weaker dependency would
# widen where they can be posted from for no one's benefit.
#
# No response on any of these returns a stored secret: the registry hands back
# `public_server_view`, which reduces `env` and `headers` to key names.


@router.get("/mcp-servers", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_mcp_servers() -> dict[str, Any]:
    """List the MCP servers registered with Code Bridge, with secrets masked."""
    return {"items": mcp_registry.public_registry()}


@router.get(
    "/mcp-servers/detected",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def list_detected_mcp_servers() -> dict[str, Any]:
    """The MCP servers a workflow step can actually run on — names only.

    Wider than the registry listing above: this is the *merged* set (CLI
    configs plus Code Bridge's registry, launchable entries only), which is
    exactly what the ``mcp_tool`` runtime gate accepts. It backs the step
    schema's ``mcp-servers`` option source, so the picker and the gate can
    never disagree about which servers exist. Names and origins only — never
    the entries themselves, whose ``env``/``headers`` carry credentials.
    """
    from agent.capability_registry import detected_mcp_server_names

    return {"items": detected_mcp_server_names()}


@router.post("/mcp-servers", dependencies=[Depends(verify_api_key)], response_model=None)
async def upsert_mcp_server(payload: McpServerUpsert) -> dict[str, Any]:
    """Register or replace one MCP server.

    Rejected with 400 when the entry cannot actually be launched from what it
    declares. Storing it anyway would list a server in the app that the launch
    path silently drops, so the failure surfaces days later as a parked step
    naming the server as missing.
    """
    try:
        view = mcp_registry.upsert_server(payload.name, payload.config)
    except mcp_registry.McpRegistryError as exc:
        _audit_mcp("system.mcp_servers.upsert", payload.name, success=False, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc))
    _audit_mcp("system.mcp_servers.upsert", payload.name, success=True, status_code=200)
    return view


@router.delete("/mcp-servers/{name}", dependencies=[Depends(verify_api_key)], response_model=None)
async def delete_mcp_server(name: str) -> dict[str, Any]:
    """Remove one Code Bridge MCP registration."""
    removed = mcp_registry.remove_server(name)
    if not removed:
        _audit_mcp("system.mcp_servers.delete", name, success=False, status_code=404)
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' is not registered")
    _audit_mcp("system.mcp_servers.delete", name, success=True, status_code=200)
    return {"name": name, "deleted": True}


def _audit_mcp(operation: str, name: str, *, success: bool, status_code: int) -> None:
    """Audit with the server name only — never the entry.

    The same discipline `routes/secrets.py::_audit` applies, and for the same
    reason: the entry holds `env` and `headers`, and an audit row is a durable
    copy of whatever it is handed.
    """
    record_api_action(
        operation=operation,
        details={"name": name},
        success=success,
        status_code=status_code,
    )


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
