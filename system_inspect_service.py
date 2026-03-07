"""Wrappers for system inspection actions used by API routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_usage import fetch_claude_usage_snapshot, merge_usage_for_display
from config import get_config
from database import get_project_db, get_usage_db
from filesystem_service import get_allowed_roots, validate_accessible_path
from project_utils import (
    MAX_SCAN_DEPTH,
    collect_existing_project_state,
    parse_excluded_dirs,
    resolve_project_path,
    scan_project_candidates,
)


@dataclass(frozen=True)
class SystemInspectResult:
    """Typed result for system-inspection route responses."""

    success: bool
    status_code: int
    payload: dict[str, Any]

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize for route response helpers."""
        return self.payload


def _error_result(status_code: int, message: str) -> SystemInspectResult:
    return SystemInspectResult(
        success=False,
        status_code=status_code,
        payload={"error": message},
    )


def list_system_directories_for_current_server(path: str | None = None) -> SystemInspectResult:
    """List subdirectories for an absolute server path.

    Security: Only paths within server's accessible_folders config are allowed.
    When path is None, returns the accessible_folders list for selection.
    """
    # If no path provided, return accessible roots for selection
    if path is None or not path.strip():
        roots = get_allowed_roots()
        if len(roots) == 1:
            # Single root: browse it directly
            path = roots[0]["path"]
        else:
            # Multiple roots: return selection screen
            return SystemInspectResult(
                success=True,
                status_code=200,
                payload={
                    "current_path": "",
                    "parent_path": None,
                    "folders": roots,
                    "is_accessible_roots": True,
                },
            )

    # Validate and resolve the path
    try:
        requested_path = Path(path).expanduser()
        if not requested_path.is_absolute():
            return _error_result(400, "Path must be absolute (start with /)")
        target_path = requested_path.resolve()
    except Exception:
        return _error_result(400, "Invalid path")

    # Security: Validate path is within accessible_folders
    if not validate_accessible_path(str(target_path)):
        return _error_result(403, f"Access denied: {path} is outside accessible folders")

    if not target_path.exists():
        return _error_result(404, f"Path not found: {target_path}")

    if not target_path.is_dir():
        return _error_result(400, f"Not a directory: {target_path}")

    try:
        directories: list[dict[str, str]] = []
        for item in sorted(target_path.iterdir(), key=lambda candidate: candidate.name.lower()):
            try:
                if not item.is_dir():
                    continue
                directories.append({"name": item.name, "path": str(item)})
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError):
        return _error_result(403, f"Cannot access directory: {target_path}")

    # Compute parent_path - only if parent is still within accessible_folders
    parent = target_path.parent
    parent_path: str | None = None
    if parent != target_path and validate_accessible_path(str(parent)):
        parent_path = str(parent)

    return SystemInspectResult(
        success=True,
        status_code=200,
        payload={
            "current_path": str(target_path),
            "parent_path": parent_path,
            "folders": directories,
            "is_accessible_roots": False,
        },
    )


def list_project_candidates_for_current_server(
    root_path: str,
    *,
    exclude_dirs: str | None = None,
    max_depth: int = 1,
    project_db: Any | None = None,
) -> SystemInspectResult:
    """Scan a root directory and return candidate project folders.

    Security: Only paths within server's accessible_folders config are allowed.
    """
    resolved_root, error, status_code = resolve_project_path(root_path)
    if resolved_root is None:
        return _error_result(status_code or 400, error or "Invalid root path")

    # Security: Validate path is within accessible_folders
    if not validate_accessible_path(str(resolved_root)):
        return _error_result(403, f"Access denied: {root_path} is outside accessible folders")

    if max_depth < 0 or max_depth > MAX_SCAN_DEPTH:
        return _error_result(400, f"max_depth must be between 0 and {MAX_SCAN_DEPTH}")

    excluded = parse_excluded_dirs(exclude_dirs)
    candidates = scan_project_candidates(
        root_path=resolved_root,
        excluded_dirs=excluded,
        max_depth=max_depth,
    )

    resolved_project_db = project_db or get_project_db()
    existing_projects = resolved_project_db.get_all()
    _, existing_paths = collect_existing_project_state(existing_projects)

    enriched_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_path = candidate.get("path", "")
        registered_name = existing_paths.get(candidate_path)
        enriched_candidates.append(
            {
                **candidate,
                "registered": registered_name is not None,
                "registered_project_name": registered_name,
            }
        )

    return SystemInspectResult(
        success=True,
        status_code=200,
        payload={
            "root_path": str(resolved_root),
            "excluded_dirs": sorted(excluded),
            "candidates": enriched_candidates,
            "count": len(enriched_candidates),
        },
    )


async def get_system_usage_for_current_server(
    *,
    config: Any | None = None,
    usage_db: Any | None = None,
) -> SystemInspectResult:
    """Get rolling weekly usage summary and budget percentage."""
    resolved_config = config or get_config()
    resolved_usage_db = usage_db or get_usage_db()
    summary = resolved_usage_db.get_weekly_summary(
        budget_usd=resolved_config.weekly_budget_usd,
        window_days=resolved_config.usage_window_days,
    )
    claude_snapshot = await fetch_claude_usage_snapshot()
    merged = merge_usage_for_display(summary, claude_snapshot)
    return SystemInspectResult(success=True, status_code=200, payload=merged)
