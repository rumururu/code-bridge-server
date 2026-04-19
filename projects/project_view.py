"""Helpers for project API response shaping."""

from __future__ import annotations

import os
from typing import Any


def build_project_view(
    project: dict[str, Any],
    detected_port: int | None,
    *,
    managed: bool = False,
    include_command: bool = True,
) -> dict[str, Any]:
    """Build normalized project response including runtime dev-server status.

    Args:
        managed: True if the dev server was started by Code Bridge (not external).
                 Only managed servers can be stopped via the UI.
    """
    dev_server = project.get("dev_server") or {}
    # running = Code Bridge가 시작한 서버만 true
    dev_server_view: dict[str, Any] = {
        "port": detected_port or dev_server.get("port"),
        "running": managed,
    }
    if include_command:
        dev_server_view["command"] = dev_server.get("command")

    path_value = str(project.get("path") or "")
    path_valid = bool(path_value) and os.path.isdir(path_value)
    path_issue: str | None = None
    if not path_value:
        path_issue = "No path is registered."
    elif not path_valid:
        path_issue = (
            "The stored path does not exist on this Mac. The project may have "
            "moved, or the DB may have been copied from another Mac."
        )

    return {
        "name": project.get("name", ""),
        "path": path_value,
        "type": project.get("type", "unknown"),
        "enabled": project.get("enabled", True),
        "dev_server": dev_server_view,
        "path_valid": path_valid,
        "path_issue": path_issue,
    }
