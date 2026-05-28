"""Service wrappers for project terminal operations."""

from typing import Any

from core.base_result import BaseRouteResult
from projects.project_manager import get_project_manager
from terminal import CommandResult, get_terminal_manager

TerminalActionResult = BaseRouteResult


def _project_path(name: str) -> str | None:
    project = get_project_manager().get_project(name)
    path = project.get("path") if project else None
    return path if isinstance(path, str) and path else None


def _missing_project(name: str) -> TerminalActionResult:
    return TerminalActionResult.error(404, f"Project '{name}' not found")


def _command_result_payload(result: CommandResult) -> dict[str, Any]:
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "error": result.error,
        "timed_out": result.timed_out,
    }


async def execute_terminal_command_for_current_server(
    name: str,
    *,
    command: str,
    timeout: int = 300,
) -> TerminalActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    session = get_terminal_manager().get_session(name, project_path)
    result = await session.execute(command, timeout=timeout)
    return TerminalActionResult.ok(_command_result_payload(result))


def get_terminal_history_for_current_server(name: str) -> TerminalActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    session = get_terminal_manager().get_session(name, project_path)
    return TerminalActionResult.ok({"history": list(session.command_history)})


async def cancel_terminal_command_for_current_server(name: str) -> TerminalActionResult:
    project_path = _project_path(name)
    if project_path is None:
        return _missing_project(name)
    session = get_terminal_manager().get_session(name, project_path)
    return TerminalActionResult.ok({"cancelled": await session.cancel()})

