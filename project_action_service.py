"""Wrappers for project manager and project registry actions used by API routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from claude_session import get_session_manager
from database import get_project_db
from filesystem_service import validate_accessible_path
from project_utils import collect_existing_project_state, prepare_project_payload
from projects import get_project_manager


@dataclass(frozen=True)
class ProjectRegistryResult:
    """Typed result for project registry route responses."""

    success: bool
    status_code: int
    payload: dict[str, Any]

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize for route response helpers."""
        return self.payload


def _project_registry_error(status_code: int, message: str) -> ProjectRegistryResult:
    return ProjectRegistryResult(
        success=False,
        status_code=status_code,
        payload={"error": message},
    )


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


async def run_project_on_device_for_current_server(
    name: str,
    device_id: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Run project on device via active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return await resolved_manager.run_project_on_device(name, device_id)


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


async def build_project_flutter_web_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Build Flutter web app via active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return await resolved_manager.build_flutter_web(name)


def get_project_build_status_for_current_server(
    name: str,
    *,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Get build status via active project manager."""
    resolved_manager = _resolve_project_manager(manager)
    return resolved_manager.get_build_status(name)


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
        return _project_registry_error(
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
        return _project_registry_error(status_code or 400, error or "Invalid project")

    created = resolved_project_db.create(payload)
    return ProjectRegistryResult(success=True, status_code=200, payload=created)


def import_project_records_for_current_server(
    paths: list[str],
    *,
    project_db: Any | None = None,
) -> ProjectRegistryResult:
    """Import multiple project records from absolute paths."""
    if not paths:
        return _project_registry_error(400, "No project paths provided")

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
        except Exception as exc:
            failed.append({"path": raw_path, "reason": f"Failed to create: {exc}"})

    return ProjectRegistryResult(
        success=True,
        status_code=200,
        payload={
            "created": created,
            "skipped": skipped,
            "failed": failed,
            "summary": {
                "created": len(created),
                "skipped": len(skipped),
                "failed": len(failed),
                "requested": len(paths),
            },
        },
    )


def update_project_record_for_current_server(
    name: str,
    updates: dict[str, Any],
    *,
    project_db: Any | None = None,
) -> ProjectRegistryResult:
    """Update an existing project record."""
    resolved_project_db = _resolve_project_db(project_db)

    if not resolved_project_db.exists(name):
        return _project_registry_error(404, f"Project {name} not found")

    updated = resolved_project_db.update(name, updates)
    return ProjectRegistryResult(success=True, status_code=200, payload=updated)


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
        return _project_registry_error(404, f"Project {name} not found")

    resolved_is_running = is_dev_server_running or is_project_dev_server_running_for_current_server
    resolved_stop_dev_server = stop_dev_server or stop_project_dev_server_for_current_server

    if resolved_is_running(name):
        await resolved_stop_dev_server(name)

    deleted = resolved_project_db.delete(name)
    if not deleted:
        return _project_registry_error(500, "Failed to delete project")

    return ProjectRegistryResult(
        success=True,
        status_code=200,
        payload={"status": "deleted", "name": name},
    )


async def close_project_session_for_current_server(
    project_name: str,
    *,
    session_manager: Any | None = None,
) -> dict[str, str]:
    """Close project chat session via active session manager."""
    resolved_session_manager = session_manager or get_session_manager()
    await resolved_session_manager.close_session(project_name)
    return {"status": "closed", "project": project_name}
