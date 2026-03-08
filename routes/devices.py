"""Device and scrcpy management API routes."""

from fastapi import APIRouter, Depends

from device_action_service import (
    get_scrcpy_status_for_current_server,
    list_connected_devices_for_current_server,
    start_scrcpy_for_current_server,
    stop_scrcpy_for_current_server,
)
from .result_response import as_flagged_response
from .deps import verify_api_key, verify_api_key_or_localhost

router = APIRouter(tags=["devices"])


@router.get("/api/devices", dependencies=[Depends(verify_api_key)])
async def list_devices():
    """List connected Android devices."""
    devices = await list_connected_devices_for_current_server()
    return {"devices": devices}


@router.get("/api/scrcpy/status", dependencies=[Depends(verify_api_key_or_localhost)])
async def scrcpy_status():
    """Get ws-scrcpy server status."""
    return get_scrcpy_status_for_current_server()


@router.post("/api/scrcpy/start", dependencies=[Depends(verify_api_key_or_localhost)])
async def start_scrcpy():
    """Start ws-scrcpy server."""
    result = await start_scrcpy_for_current_server()
    return as_flagged_response(result, error_status_code=400)


@router.post("/api/scrcpy/stop", dependencies=[Depends(verify_api_key_or_localhost)])
async def stop_scrcpy():
    """Stop ws-scrcpy server."""
    result = await stop_scrcpy_for_current_server()
    return as_flagged_response(result, error_status_code=400)
