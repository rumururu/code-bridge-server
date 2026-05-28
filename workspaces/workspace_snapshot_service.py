"""Workspace filesystem snapshot helpers."""

from pathlib import Path
from typing import Any

from audit.route_audit import record_api_action

from .workspace_store import get_workspace_store

EXCLUDED_NAMES = {
    ".git",
    ".next",
    ".dart_tool",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
}


def build_workspace_snapshot(
    workspace_id: str,
    *,
    limit: int = 200,
) -> dict[str, Any] | None:
    """Build a shallow workspace snapshot for agent planning."""
    workspace = get_workspace_store().get_workspace(workspace_id)
    if workspace is None:
        return None

    root_path = Path(workspace["root_path"]).expanduser()
    entries: list[dict[str, Any]] = []
    root_exists = root_path.exists()
    root_is_dir = root_path.is_dir()

    if root_exists and root_is_dir:
        for item in sorted(root_path.iterdir(), key=lambda path: path.name.lower()):
            if item.name in EXCLUDED_NAMES:
                continue
            try:
                stat = item.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "kind": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified_at": stat.st_mtime,
                }
            )
            if len(entries) >= max(1, min(int(limit), 500)):
                break

    snapshot = {
        "workspace": workspace,
        "root_exists": root_exists,
        "root_is_dir": root_is_dir,
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "file_count": sum(1 for entry in entries if entry["kind"] == "file"),
            "directory_count": sum(1 for entry in entries if entry["kind"] == "directory"),
        },
    }
    record_api_action(
        operation="workspace.snapshot",
        project_name=workspace.get("project_name"),
        details={
            "workspace_id": workspace_id,
            "root_path": workspace.get("root_path"),
            "entry_count": len(entries),
        },
        success=True,
        status_code=200,
    )
    return snapshot
