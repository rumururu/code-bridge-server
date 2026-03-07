"""Preparation helpers for starting project dev servers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from project_dev_server import infer_default_dev_server_command_from_project
from project_models import ProjectType

NO_COMMAND_TEMPLATE = (
    "No dev server command configured for {name}. "
    "Run the project's dev server manually and press refresh, "
    "or configure a command."
)


@dataclass(frozen=True)
class DevServerStartPlan:
    """Resolved start inputs for one project dev server."""

    command: str
    project_path: str
    project_type: ProjectType
    port_hint: int | None


@dataclass(frozen=True)
class DevServerStartPlanResult:
    """Result of preparing a dev-server start."""

    success: bool
    plan: DevServerStartPlan | None = None
    error_message: str | None = None


def resolve_dev_server_start_plan(
    name: str,
    project: dict[str, Any],
    *,
    project_db: Any | None = None,
    infer_command: Callable[[str, Any], str | None] = infer_default_dev_server_command_from_project,
) -> DevServerStartPlanResult:
    """Resolve and validate command/path/type before starting dev server."""
    dev_server = project.get("dev_server") or {}
    command = dev_server.get("command")
    port = dev_server.get("port")
    project_path = project.get("path")
    project_type = ProjectType.from_string(project.get("type", ""))

    if not project_path:
        return DevServerStartPlanResult(
            success=False,
            error_message=f"Invalid dev server configuration for {name}",
        )

    if not isinstance(command, str) or not command.strip():
        command = infer_command(str(project_path), project_type)
        if command and project_db is not None:
            try:
                project_db.update(name, {"dev_server": {"command": command}})
            except Exception:
                pass

    if not command:
        return DevServerStartPlanResult(
            success=False,
            error_message=NO_COMMAND_TEMPLATE.format(name=name),
        )

    if not Path(project_path).exists():
        return DevServerStartPlanResult(
            success=False,
            error_message=f"Project path does not exist: {project_path}",
        )

    return DevServerStartPlanResult(
        success=True,
        plan=DevServerStartPlan(
            command=command,
            project_path=str(project_path),
            project_type=project_type,
            port_hint=port if isinstance(port, int) else None,
        ),
    )


def spawn_dev_server_process(command: str, project_path: str) -> subprocess.Popen:
    """Spawn one shell-based dev-server process."""
    return subprocess.Popen(
        command,
        shell=True,
        cwd=project_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
