"""Wrappers for project file actions used by API routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

from files import get_file_manager
from projects import get_project_manager


FileManagerFactory = Callable[[str], Any]
MAX_ATTACHMENT_UPLOAD_BYTES = 15 * 1024 * 1024
ATTACHMENTS_DIR_NAME = ".codebridge_uploads"


@dataclass(frozen=True)
class FileActionResult:
    """Typed result for file-route responses."""

    success: bool
    status_code: int
    payload: dict[str, Any]

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize for route response helpers."""
        return self.payload


def _resolve_project_file_manager(
    project_name: str,
    *,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> tuple[Any | None, FileActionResult | None]:
    resolved_manager = manager or get_project_manager()
    project = resolved_manager.get_project(project_name)

    if project is None:
        return None, FileActionResult(
            success=False,
            status_code=404,
            payload={"error": f"Project {project_name} not found"},
        )

    project_path = project.get("path")
    if not project_path:
        return None, FileActionResult(
            success=False,
            status_code=400,
            payload={"error": "Project has no path configured"},
        )

    resolved_factory = file_manager_factory or get_file_manager
    return resolved_factory(project_path), None


def _resolve_project_root_path(
    project_name: str,
    *,
    manager: Any | None = None,
) -> tuple[Path | None, FileActionResult | None]:
    resolved_manager = manager or get_project_manager()
    project = resolved_manager.get_project(project_name)

    if project is None:
        return None, FileActionResult(
            success=False,
            status_code=404,
            payload={"error": f"Project {project_name} not found"},
        )

    project_path = project.get("path")
    if not project_path:
        return None, FileActionResult(
            success=False,
            status_code=400,
            payload={"error": "Project has no path configured"},
        )

    try:
        root = Path(project_path).resolve()
    except OSError:
        return None, FileActionResult(
            success=False,
            status_code=400,
            payload={"error": "Project path is invalid"},
        )

    if not root.exists() or not root.is_dir():
        return None, FileActionResult(
            success=False,
            status_code=400,
            payload={"error": "Project path does not exist"},
        )

    return root, None


def _to_file_action_result(payload: dict[str, Any]) -> FileActionResult:
    if "error" in payload:
        return FileActionResult(
            success=False,
            status_code=int(payload.get("code", 400)),
            payload={"error": str(payload["error"])},
        )
    return FileActionResult(success=True, status_code=200, payload=payload)


def _safe_attachment_filename(original_name: str) -> str:
    name = Path(original_name or "").name.strip()
    if not name:
        name = "attachment"

    stem = Path(name).stem or "attachment"
    suffix = Path(name).suffix
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = "attachment"
    if len(safe_stem) > 80:
        safe_stem = safe_stem[:80]

    if len(suffix) > 16:
        suffix = suffix[:16]

    return f"{safe_stem}{suffix}"


def _run_operation_for_project(
    project_name: str,
    operation: Callable[[Any], dict[str, Any]],
    *,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    file_manager, resolution_error = _resolve_project_file_manager(
        project_name,
        manager=manager,
        file_manager_factory=file_manager_factory,
    )
    if resolution_error is not None:
        return resolution_error
    return _to_file_action_result(operation(file_manager))


def list_project_files_for_current_server(
    project_name: str,
    path: str = "",
    *,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """List files in a project directory."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.list_directory(path),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def read_project_file_for_current_server(
    project_name: str,
    path: str,
    *,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Read one file from a project."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.read_file(path),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def write_project_file_content_for_current_server(
    project_name: str,
    path: str,
    content: str,
    *,
    create_dirs: bool = False,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Write file content in a project."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.write_file(path, content, create_dirs),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def create_project_file_or_directory_for_current_server(
    project_name: str,
    path: str,
    *,
    is_directory: bool = False,
    content: str = "",
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Create a file or directory in a project."""
    if is_directory:
        operation = lambda file_manager: file_manager.create_directory(path)
    else:
        operation = lambda file_manager: file_manager.write_file(path, content, create_dirs=True)

    return _run_operation_for_project(
        project_name,
        operation,
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def delete_project_path_for_current_server(
    project_name: str,
    path: str,
    *,
    recursive: bool = False,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Delete a file (or directory when recursive fallback is enabled)."""
    file_manager, resolution_error = _resolve_project_file_manager(
        project_name,
        manager=manager,
        file_manager_factory=file_manager_factory,
    )
    if resolution_error is not None:
        return resolution_error

    payload = file_manager.delete_file(path)
    if "error" in payload and "Not a file" in str(payload.get("error", "")) and recursive:
        payload = file_manager.delete_directory(path)
    return _to_file_action_result(payload)


def rename_project_path_for_current_server(
    project_name: str,
    old_path: str,
    new_path: str,
    *,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Rename or move a file/directory in a project."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.rename_file(old_path, new_path),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def copy_project_path_for_current_server(
    project_name: str,
    source: str,
    dest: str,
    *,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Copy file/directory in a project."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.copy_file(source, dest),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def move_project_path_for_current_server(
    project_name: str,
    source: str,
    dest: str,
    *,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Move file/directory in a project."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.rename_file(source, dest),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def search_project_files_for_current_server(
    project_name: str,
    query: str,
    *,
    limit: int = 50,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Search files by name within a project."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.search_files(query, limit=limit),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def search_project_file_content_for_current_server(
    project_name: str,
    query: str,
    *,
    limit: int = 100,
    case_sensitive: bool = False,
    manager: Any | None = None,
    file_manager_factory: FileManagerFactory | None = None,
) -> FileActionResult:
    """Search file contents within a project."""
    return _run_operation_for_project(
        project_name,
        lambda file_manager: file_manager.search_content(
            query,
            limit=limit,
            case_sensitive=case_sensitive,
        ),
        manager=manager,
        file_manager_factory=file_manager_factory,
    )


def upload_project_attachment_for_current_server(
    project_name: str,
    *,
    filename: str,
    content: bytes,
    content_type: str | None = None,
    source: str = "file",
    manager: Any | None = None,
) -> FileActionResult:
    """Store an uploaded attachment in the project workspace.

    The uploaded binary is written into `.codebridge_uploads/` inside the
    target project so the LLM session can reference it with project-relative
    paths.
    """
    project_root, resolution_error = _resolve_project_root_path(
        project_name,
        manager=manager,
    )
    if resolution_error is not None:
        return resolution_error

    file_size = len(content)
    if file_size <= 0:
        return FileActionResult(
            success=False,
            status_code=400,
            payload={"error": "Uploaded file is empty"},
        )

    if file_size > MAX_ATTACHMENT_UPLOAD_BYTES:
        return FileActionResult(
            success=False,
            status_code=413,
            payload={
                "error": (
                    f"File too large ({file_size} bytes). "
                    f"Max: {MAX_ATTACHMENT_UPLOAD_BYTES} bytes"
                )
            },
        )

    safe_name = _safe_attachment_filename(filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid4().hex[:8]
    stored_name = f"{timestamp}-{unique_suffix}-{safe_name}"

    uploads_dir = (project_root / ATTACHMENTS_DIR_NAME).resolve()
    try:
        uploads_dir.relative_to(project_root)
    except ValueError:
        return FileActionResult(
            success=False,
            status_code=400,
            payload={"error": "Invalid upload directory"},
        )

    try:
        uploads_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return FileActionResult(
            success=False,
            status_code=500,
            payload={"error": f"Cannot create upload directory: {exc}"},
        )

    stored_path = uploads_dir / stored_name
    try:
        stored_path.write_bytes(content)
    except OSError as exc:
        return FileActionResult(
            success=False,
            status_code=500,
            payload={"error": f"Cannot store uploaded file: {exc}"},
        )

    relative_path = stored_path.relative_to(project_root).as_posix()
    payload = {
        "success": True,
        "path": relative_path,
        "name": safe_name,
        "stored_name": stored_name,
        "size": file_size,
        "content_type": content_type or "application/octet-stream",
        "source": source,
    }
    return FileActionResult(success=True, status_code=200, payload=payload)
