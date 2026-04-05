"""System remote access routes: network, Firebase auth, and tunnel control."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from remote.remote_access_service import (
    build_remote_network_status_for_current_server,
    disconnect_remote_access_for_current_server,
    login_for_remote_access_request_json_for_current_server,
    logout_remote_access_for_current_server,
    start_tunnel_for_current_server,
    stop_tunnel_for_current_server,
)
from .result_response import as_route_response
from .deps import verify_api_key, verify_api_key_or_localhost

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/network-status", dependencies=[Depends(verify_api_key_or_localhost)])
async def get_network_status() -> dict[str, Any]:
    """Get network discovery and remote access status."""
    return build_remote_network_status_for_current_server().as_response_fields()


@router.post("/remote-access/login", dependencies=[Depends(verify_api_key)], response_model=None)
async def remote_access_login(request: Request) -> dict[str, Any] | Response:
    """Authenticate server with Firebase ID token from app."""
    result = await login_for_remote_access_request_json_for_current_server(request.json)
    return as_route_response(result)


@router.post("/remote-access/logout", dependencies=[Depends(verify_api_key)], response_model=None)
async def remote_access_logout() -> dict[str, Any] | Response:
    """Sign out from remote access."""
    result = await logout_remote_access_for_current_server()
    return as_route_response(result)


@router.post("/remote-access/disconnect", dependencies=[Depends(verify_api_key)], response_model=None)
async def remote_access_disconnect() -> dict[str, Any] | Response:
    """Completely disconnect from remote access."""
    result = await disconnect_remote_access_for_current_server()
    return as_route_response(result)


@router.post("/tunnel/start", dependencies=[Depends(verify_api_key_or_localhost)], response_model=None)
async def start_tunnel() -> dict[str, Any] | Response:
    """Manually start Cloudflare Tunnel."""
    result = await start_tunnel_for_current_server()
    return as_route_response(result)


@router.post("/tunnel/stop", dependencies=[Depends(verify_api_key_or_localhost)], response_model=None)
async def stop_tunnel() -> dict[str, Any] | Response:
    """Stop Cloudflare Tunnel."""
    result = await stop_tunnel_for_current_server()
    return as_route_response(result)
