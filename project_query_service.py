"""Helpers for project view queries and runtime port detection."""

from __future__ import annotations

from typing import Any, Callable

from project_models import ProjectType
from project_runtime_helpers import is_local_port_open
from project_view import build_project_view


def build_project_list_view(
    projects: list[dict[str, Any]],
    *,
    get_server_port: Callable[[str], int | None],
) -> list[dict[str, Any]]:
    """Build list response for all projects with runtime status."""
    result: list[dict[str, Any]] = []
    for project in projects:
        name = project.get("name", "")
        detected_port = get_server_port(name)
        result.append(
            build_project_view(
                project,
                detected_port,
                include_command=False,
            )
        )
    return result


def build_single_project_view(
    name: str,
    project: dict[str, Any] | None,
    *,
    get_server_port: Callable[[str], int | None],
) -> dict[str, Any] | None:
    """Build detail response for one project or return None when missing."""
    if project is None:
        return None
    detected_port = get_server_port(name)
    return build_project_view(project, detected_port)


def detect_project_running_server_port(
    project: dict[str, Any] | None,
    *,
    detect_port_for_project: Callable[[str, ProjectType], int | None],
) -> int | None:
    """Detect active dev-server port from runtime or configured fallback."""
    if not project:
        return None

    project_path = project.get("path")
    if not project_path:
        return None

    project_type = ProjectType.from_string(project.get("type", ""))
    detected_port = detect_port_for_project(str(project_path), project_type)
    if detected_port is not None:
        return detected_port

    configured_port = (project.get("dev_server") or {}).get("port")
    if isinstance(configured_port, int) and is_local_port_open(configured_port):
        return configured_port
    return None
