"""Helpers for project API response shaping."""

from __future__ import annotations

from typing import Any


def build_project_view(
    project: dict[str, Any],
    detected_port: int | None,
    *,
    include_command: bool = True,
) -> dict[str, Any]:
    """Build normalized project response including runtime dev-server status."""
    dev_server = project.get("dev_server") or {}
    is_running = detected_port is not None
    dev_server_view: dict[str, Any] = {
        "port": detected_port or dev_server.get("port"),
        "running": is_running,
    }
    if include_command:
        dev_server_view["command"] = dev_server.get("command")

    return {
        "name": project.get("name", ""),
        "path": project.get("path", ""),
        "type": project.get("type", "unknown"),
        "enabled": project.get("enabled", True),
        "dev_server": dev_server_view,
    }
