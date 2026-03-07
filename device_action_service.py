"""Wrappers for scrcpy/device manager actions used by API routes."""

from __future__ import annotations

from typing import Any

from scrcpy_manager import get_scrcpy_manager


async def list_connected_devices_for_current_server(
    *,
    scrcpy_manager: Any | None = None,
) -> list[dict[str, Any]]:
    """List connected devices from active scrcpy manager."""
    resolved_scrcpy = scrcpy_manager or get_scrcpy_manager()
    return await resolved_scrcpy.get_devices()


def get_scrcpy_status_for_current_server(
    *,
    scrcpy_manager: Any | None = None,
) -> dict[str, Any]:
    """Get scrcpy status from active scrcpy manager."""
    resolved_scrcpy = scrcpy_manager or get_scrcpy_manager()
    return resolved_scrcpy.get_status()


async def start_scrcpy_for_current_server(
    *,
    scrcpy_manager: Any | None = None,
) -> dict[str, Any]:
    """Start scrcpy via active scrcpy manager."""
    resolved_scrcpy = scrcpy_manager or get_scrcpy_manager()
    return await resolved_scrcpy.start()


async def stop_scrcpy_for_current_server(
    *,
    scrcpy_manager: Any | None = None,
) -> dict[str, Any]:
    """Stop scrcpy via active scrcpy manager."""
    resolved_scrcpy = scrcpy_manager or get_scrcpy_manager()
    return await resolved_scrcpy.stop()
