"""Device and scrcpy management API routes."""

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from fastapi.responses import Response

from devices.avd_install_manager import (
    DEFAULT_AVD_NAME,
    DEFAULT_DEVICE_PROFILE,
    DEFAULT_IMAGE,
    get_avd_install_manager,
)
from devices.device_action_service import (
    get_scrcpy_status_for_current_server,
    list_connected_devices_for_current_server,
    start_scrcpy_for_current_server,
    stop_scrcpy_for_current_server,
)
from devices.scrcpy_manager import get_scrcpy_manager
from auth.auth_service import validate_api_key_for_current_server
from .result_response import as_flagged_response
from .deps import verify_api_key, verify_api_key_or_localhost

logger = logging.getLogger(__name__)

router = APIRouter(tags=["devices"])


@router.get("/api/devices", dependencies=[Depends(verify_api_key)])
async def list_devices() -> dict[str, Any]:
    """List connected Android devices."""
    devices = await list_connected_devices_for_current_server()
    return {"devices": devices}


@router.get("/api/scrcpy/status", dependencies=[Depends(verify_api_key_or_localhost)])
async def scrcpy_status() -> dict[str, Any]:
    """Get ws-scrcpy server status."""
    return get_scrcpy_status_for_current_server()


@router.post("/api/scrcpy/start", dependencies=[Depends(verify_api_key_or_localhost)], response_model=None)
async def start_scrcpy() -> dict[str, Any] | Response:
    """Start ws-scrcpy server."""
    result = await start_scrcpy_for_current_server()
    return as_flagged_response(result, error_status_code=400)


@router.post("/api/scrcpy/stop", dependencies=[Depends(verify_api_key_or_localhost)], response_model=None)
async def stop_scrcpy() -> dict[str, Any] | Response:
    """Stop ws-scrcpy server."""
    result = await stop_scrcpy_for_current_server()
    return as_flagged_response(result, error_status_code=400)


@router.post(
    "/api/devices/{device_id}/display-override",
    dependencies=[Depends(verify_api_key)],
)
async def configure_display_override(
    device_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Apply wm size / wm density override on a running emulator without
    re-launching Chrome. Used to reflect live preview preset changes."""
    width = payload.get("width")
    height = payload.get("height")
    density = payload.get("density")
    reset_to_default = bool(payload.get("reset_to_default", False))
    scrcpy_manager = get_scrcpy_manager()
    error = await scrcpy_manager.configure_emulator_display(
        device_id,
        width=width,
        height=height,
        density=density,
        reset_to_default=reset_to_default,
    )
    if error:
        raise HTTPException(status_code=400, detail=error)
    return {
        "success": True,
        "device_id": device_id,
        "width": width,
        "height": height,
        "density": density,
        "reset_to_default": reset_to_default,
    }


# ---------- AVD install ----------

@router.get(
    "/api/devices/install-avd/environment",
    dependencies=[Depends(verify_api_key)],
)
async def install_avd_environment() -> dict[str, Any]:
    """Expose Android SDK tool paths so the app can pre-flight."""
    manager = get_avd_install_manager()
    env = manager.environment_status()
    return {
        **env,
        "ready": bool(env["sdkmanager"] and env["avdmanager"]),
        "default_image": DEFAULT_IMAGE,
        "default_device_profile": DEFAULT_DEVICE_PROFILE,
        "default_avd_name": DEFAULT_AVD_NAME,
    }


@router.post(
    "/api/devices/install-avd",
    dependencies=[Depends(verify_api_key)],
)
async def start_avd_install(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Kick off an AVD install job. Returns a job id for progress subscription."""
    manager = get_avd_install_manager()
    env = manager.environment_status()
    if not env["sdkmanager"] or not env["avdmanager"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "sdk_tools_missing",
                "message": (
                    "Android SDK cmdline-tools not found. "
                    "Install Android Studio or sdkmanager first."
                ),
                **env,
            },
        )
    avd_name = (payload.get("avd_name") or DEFAULT_AVD_NAME).strip() or DEFAULT_AVD_NAME
    image = (payload.get("image") or DEFAULT_IMAGE).strip() or DEFAULT_IMAGE
    device_profile = (
        (payload.get("device_profile") or DEFAULT_DEVICE_PROFILE).strip()
        or DEFAULT_DEVICE_PROFILE
    )
    job = await manager.start_install(
        avd_name=avd_name, image=image, device_profile=device_profile
    )
    return manager.serialize(job)


@router.get(
    "/api/devices/install-avd/{job_id}",
    dependencies=[Depends(verify_api_key)],
)
async def get_avd_install_status(job_id: str) -> dict[str, Any]:
    manager = get_avd_install_manager()
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return manager.serialize(job)


@router.websocket("/ws/devices/install-avd/{job_id}")
async def avd_install_ws(
    websocket: WebSocket,
    job_id: str,
    api_key: Optional[str] = Query(None),
) -> None:
    """Stream realtime install-progress events until the job finishes."""
    auth = validate_api_key_for_current_server(api_key)
    if not auth.success:
        await websocket.close(code=4401, reason="unauthorized")
        return

    manager = get_avd_install_manager()
    job = manager.get_job(job_id)
    if job is None:
        await websocket.close(code=4404, reason="job not found")
        return

    await websocket.accept()
    queue: Optional[asyncio.Queue] = None
    try:
        queue = await manager.subscribe(job_id)
        while True:
            event = await queue.get()
            if event is None:
                break
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("avd install ws disconnected job=%s", job_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("avd install ws crashed job=%s: %s", job_id, exc)
    finally:
        if queue is not None:
            await manager.unsubscribe(job_id, queue)
