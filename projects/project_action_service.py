"""Wrappers for project manager and project registry actions used by API routes."""

from __future__ import annotations

import sqlite3
from typing import Any, Awaitable, Callable

from llm.claude_session import get_session_manager
from core.database import get_project_db
from core.base_result import BaseRouteResult
from files.filesystem_service import validate_accessible_path
from .project_utils import collect_existing_project_state, prepare_project_payload
from .project_utils import resolve_project_path, sanitize_project_name
from projects.projects import get_project_manager

# Backwards-compatible alias
ProjectRegistryResult = BaseRouteResult


def _resolve_project_manager(manager: Any | None) -> Any:
    return manager or get_project_manager()


def _resolve_project_db(project_db: Any | None) -> Any:
    return project_db or get_project_db()


def _load_existing_project_state(project_db: Any) -> tuple[set[str], dict[str, str]]:
    existing_projects = project_db.get_all()
    return collect_existing_project_state(existing_projects)


def list_projects_for_current_server(*, manager: Any | None = None) -> list[dict[str, Any]]:
    """List configured projects from the active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return resolved_manager.get_all_projects()


def get_project_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any] | None:
    """Get one project view from the active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return resolved_manager.get_project(name)


def is_project_dev_server_running_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> bool:
    """Return whether a project's dev server is currently running."""
    resolved_manager = _resolve_project_manager(manager)
    return bool(resolved_manager.is_server_running(name))


async def start_project_dev_server_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Start a project's dev server via active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return await resolved_manager.start_dev_server(name)


async def stop_project_dev_server_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Stop a project's dev server via active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return await resolved_manager.stop_dev_server(name)


async def restart_project_dev_server_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Restart a project's dev server (stop then start)."""
    resolved_manager = _resolve_project_manager(manager)

    # Stop first (ignore errors if not running)
    stop_result = await resolved_manager.stop_dev_server(name)

    # Start the server
    start_result = await resolved_manager.start_dev_server(name)

    # Check success based on start_result
    success = start_result.get("success", False)

    return {
        "success": success,
        "status": "restarted" if success else "restart_failed",
        "stop_result": stop_result,
        "start_result": start_result,
    }


async def run_project_on_device_for_current_server(
    name: str,
    device_id: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Run project on device via active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return await resolved_manager.run_project_on_device(name, device_id)


async def open_web_preview_on_device_for_current_server(
    name: str,
    device_id: str,
    *,
    width: int | None = None,
    height: int | None = None,
    density: int | None = None,
    reset_to_default: bool = False,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Open a web project's dev server preview on an Android emulator."""
    resolved_manager = _resolve_project_manager(manager)
    return await resolved_manager.open_web_preview_on_device(
        name,
        device_id,
        width=width,
        height=height,
        density=density,
        reset_to_default=reset_to_default,
    )


async def stop_project_device_run_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Stop project device run via active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return await resolved_manager.stop_project_on_device(name)


def get_project_device_run_log_for_current_server(
    name: str,
    *,
    lines: int = 120,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Get project's device run log from active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return resolved_manager.get_device_run_log(name, lines=lines)


def create_project_record_for_current_server(
    *,
    path_value: str,
    requested_name: str | None = None,
    requested_type: str | None = None,
    dev_server: dict[str, Any] | None = None,
    project_db: Any | None = None,
) -> ProjectRegistryResult:
    """Create a project record using project-path validation helpers."""
    # Validate path is within accessible_folders (security boundary)
    if not validate_accessible_path(path_value):
        return ProjectRegistryResult.error(
            403,
            f"Path '{path_value}' is outside accessible folders. "
            "Add the parent folder to Accessible Folders first.",
        )

    resolved_project_db = _resolve_project_db(project_db)
    existing_names, existing_paths = _load_existing_project_state(resolved_project_db)

    payload, error, status_code = prepare_project_payload(
        path_value=path_value,
        existing_names=existing_names,
        existing_paths=existing_paths,
        requested_name=requested_name,
        requested_type=requested_type,
        dev_server=dev_server,
    )
    if payload is None:
        return ProjectRegistryResult.error(status_code or 400, error or "Invalid project")

    created = resolved_project_db.create(payload)
    return ProjectRegistryResult.ok(created)


def create_project_folder_for_current_server(
    *,
    root_path: str,
    folder_name: str,
    requested_name: str | None = None,
    requested_type: str | None = None,
    dev_server: dict[str, Any] | None = None,
    project_db: Any | None = None,
) -> ProjectRegistryResult:
    """Create a folder inside a root path and register it as a project."""
    resolved_root, error, status_code = resolve_project_path(root_path)
    if resolved_root is None:
        return ProjectRegistryResult.error(status_code or 400, error or "Invalid root path")

    if not validate_accessible_path(str(resolved_root)):
        return ProjectRegistryResult.error(
            403,
            f"Root path '{resolved_root}' is outside accessible folders. "
            "Add the parent folder to Accessible Folders first.",
        )

    sanitized_folder_name = sanitize_project_name(folder_name)
    if sanitized_folder_name in {"", ".", ".."}:
        return ProjectRegistryResult.error(400, "Project folder name is required")

    project_path = (resolved_root / sanitized_folder_name).resolve()
    try:
        project_path.relative_to(resolved_root)
    except ValueError:
        return ProjectRegistryResult.error(400, "Invalid project folder name")

    if project_path.exists():
        return ProjectRegistryResult.error(409, f"Project folder already exists: {project_path}")

    resolved_project_db = _resolve_project_db(project_db)
    existing_names, existing_paths = _load_existing_project_state(resolved_project_db)

    try:
        project_path.mkdir()
    except OSError as exc:
        return ProjectRegistryResult.error(500, f"Failed to create project folder: {exc}")

    payload, payload_error, payload_status_code = prepare_project_payload(
        path_value=str(project_path),
        existing_names=existing_names,
        existing_paths=existing_paths,
        requested_name=requested_name or sanitized_folder_name,
        requested_type=requested_type,
        dev_server=dev_server,
        skip_accessible_check=True,
    )
    if payload is None:
        try:
            project_path.rmdir()
        except OSError:
            pass
        return ProjectRegistryResult.error(
            payload_status_code or 400,
            payload_error or "Invalid project",
        )

    try:
        created = resolved_project_db.create(payload)
    except (sqlite3.Error, KeyError, TypeError) as exc:
        try:
            project_path.rmdir()
        except OSError:
            pass
        return ProjectRegistryResult.error(500, f"Failed to register project: {exc}")

    return ProjectRegistryResult.ok(created, status_code=201)


def import_project_records_for_current_server(
    paths: list[str],
    *,
    project_db: Any | None = None,
) -> ProjectRegistryResult:
    """Import multiple project records from absolute paths."""
    if not paths:
        return ProjectRegistryResult.error(400, "No project paths provided")

    resolved_project_db = _resolve_project_db(project_db)
    existing_names, existing_paths = _load_existing_project_state(resolved_project_db)

    created: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for raw_path in paths:
        # Validate path is within accessible_folders (security boundary)
        if not validate_accessible_path(raw_path):
            failed.append({
                "path": raw_path,
                "reason": "Path is outside accessible folders",
            })
            continue

        payload, error, status_code = prepare_project_payload(
            path_value=raw_path,
            existing_names=existing_names,
            existing_paths=existing_paths,
        )

        if payload is None:
            item = {"path": raw_path, "reason": error or "Invalid project path"}
            if status_code == 400 and "already registered as project" in (error or ""):
                skipped.append(item)
            else:
                failed.append(item)
            continue

        try:
            created_project = resolved_project_db.create(payload)
            created.append(created_project)
            created_name = created_project.get("name")
            created_path = created_project.get("path")
            if isinstance(created_name, str):
                existing_names.add(created_name)
            if isinstance(created_path, str):
                existing_paths[created_path] = created_name or ""
        except (sqlite3.Error, KeyError, TypeError) as exc:
            failed.append({"path": raw_path, "reason": f"Failed to create: {exc}"})

    return ProjectRegistryResult.ok({
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "summary": {
            "created": len(created),
            "skipped": len(skipped),
            "failed": len(failed),
            "requested": len(paths),
        },
    })


def update_project_record_for_current_server(
    name: str,
    updates: dict[str, Any],
    *,
    project_db: Any | None = None,
) -> ProjectRegistryResult:
    """Update an existing project record."""
    resolved_project_db = _resolve_project_db(project_db)

    if not resolved_project_db.exists(name):
        return ProjectRegistryResult.error(404, f"Project {name} not found")

    updated = resolved_project_db.update(name, updates)
    return ProjectRegistryResult.ok(updated)


async def delete_project_record_for_current_server(
    name: str,
    *,
    project_db: Any | None = None,
    is_dev_server_running: Callable[[str], bool] | None = None,
    stop_dev_server: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
) -> ProjectRegistryResult:
    """Delete a project and stop dev server first when needed."""
    resolved_project_db = _resolve_project_db(project_db)

    if not resolved_project_db.exists(name):
        return ProjectRegistryResult.error(404, f"Project {name} not found")

    resolved_is_running = is_dev_server_running or is_project_dev_server_running_for_current_server
    resolved_stop_dev_server = stop_dev_server or stop_project_dev_server_for_current_server

    if resolved_is_running(name):
        await resolved_stop_dev_server(name)

    deleted = resolved_project_db.delete(name)
    if not deleted:
        return ProjectRegistryResult.error(500, "Failed to delete project")

    return ProjectRegistryResult.ok({"status": "deleted", "name": name})


async def close_project_session_for_current_server(
    project_name: str,
    *,
    session_manager: Any | None = None,
) -> dict[str, str]:
    """Close project chat session via active session manager."""
    resolved_session_manager = session_manager or get_session_manager()
    await resolved_session_manager.close_session(project_name)
    return {"status": "closed", "project": project_name}
