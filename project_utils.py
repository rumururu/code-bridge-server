"""Project path validation and candidate scanning helpers."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Optional

NEXT_CONFIG_MARKERS = (
    "next.config.js",
    "next.config.mjs",
    "next.config.ts",
    "next.config.cjs",
)

DEFAULT_SCAN_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".dart_tool",
    ".next",
    "node_modules",
    "build",
    "dist",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

MAX_SCAN_DEPTH = 5
MAX_SCAN_RESULTS = 300


def sanitize_project_name(value: str) -> str:
    """Sanitize user-provided project name into a safe identifier."""
    sanitized = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "project"


def build_unique_project_name(base_name: str, existing_names: set[str]) -> str:
    """Create a unique project name by adding a numeric suffix if needed."""
    if base_name not in existing_names:
        return base_name

    index = 2
    while f"{base_name}_{index}" in existing_names:
        index += 1
    return f"{base_name}_{index}"


def infer_project_type(project_path: Path) -> str:
    """Infer project type from common marker files."""
    if (project_path / "pubspec.yaml").exists():
        return "flutter"

    package_json = project_path / "package.json"
    if package_json.exists():
        try:
            package_data = json.loads(package_json.read_text(encoding="utf-8"))
            dependencies = package_data.get("dependencies", {}) or {}
            dev_dependencies = package_data.get("devDependencies", {}) or {}
            if "next" in dependencies or "next" in dev_dependencies:
                return "nextjs"
        except Exception:
            pass
        return "web"

    for marker in NEXT_CONFIG_MARKERS:
        if (project_path / marker).exists():
            return "nextjs"

    return "other"


def normalize_project_type(value: Optional[str], project_path: Path) -> str:
    """Normalize provided project type or infer it from marker files."""
    if value is None or not value.strip():
        return infer_project_type(project_path)

    normalized = value.strip().lower()
    if normalized in {"next", "next.js"}:
        return "nextjs"
    return normalized


def resolve_project_path(path_value: str) -> tuple[Path | None, str | None, int | None]:
    """Resolve and validate a project path."""
    raw_path = path_value.strip()
    if not raw_path:
        return None, "Project path is required", 400

    requested_path = Path(raw_path).expanduser()
    if not requested_path.is_absolute():
        return None, "Project path must be absolute (start with /)", 400

    try:
        resolved_path = requested_path.resolve()
    except Exception:
        return None, "Invalid project path", 400

    if not resolved_path.exists():
        return None, f"Project path not found: {resolved_path}", 404

    if not resolved_path.is_dir():
        return None, f"Not a directory: {resolved_path}", 400

    return resolved_path, None, None


def collect_existing_project_state(
    projects: list[dict[str, Any]],
) -> tuple[set[str], dict[str, str]]:
    """Build lookup maps for existing project names and paths."""
    existing_names: set[str] = set()
    existing_paths: dict[str, str] = {}

    for project in projects:
        project_name = str(project.get("name", "")).strip()
        if project_name:
            existing_names.add(project_name)

        project_path = project.get("path")
        if not isinstance(project_path, str):
            continue

        try:
            resolved = str(Path(project_path).expanduser().resolve())
            existing_paths[resolved] = project_name or resolved
        except Exception:
            continue

    return existing_names, existing_paths


def prepare_project_payload(
    path_value: str,
    existing_names: set[str],
    existing_paths: dict[str, str],
    requested_name: Optional[str] = None,
    requested_type: Optional[str] = None,
    dev_server: Optional[dict[str, Any]] = None,
    skip_accessible_check: bool = False,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """Validate input and build DB payload for a project.

    Args:
        path_value: Absolute path to the project directory.
        existing_names: Set of already used project names.
        existing_paths: Map of resolved paths to their project names.
        requested_name: Optional custom project name.
        requested_type: Optional project type override.
        dev_server: Optional dev server configuration.
        skip_accessible_check: If True, skip accessible_folders validation.

    Returns:
        Tuple of (payload, error, status_code). On success, error and status_code are None.
    """
    resolved_path, error, status_code = resolve_project_path(path_value)
    if resolved_path is None:
        return None, error, status_code

    # Validate path is within accessible_folders
    if not skip_accessible_check:
        from filesystem_service import validate_accessible_path

        if not validate_accessible_path(str(resolved_path)):
            return (
                None,
                f"Path is outside accessible folders: {resolved_path}. "
                "Add the parent folder to Accessible Folders first.",
                403,
            )

    resolved_key = str(resolved_path)
    existing_owner = existing_paths.get(resolved_key)
    if existing_owner is not None:
        return None, f"Path already registered as project {existing_owner}", 400

    if requested_name is None or not requested_name.strip():
        base_name = sanitize_project_name(resolved_path.name)
        project_name = build_unique_project_name(base_name, existing_names)
    else:
        project_name = sanitize_project_name(requested_name)
        if project_name in existing_names:
            return None, f"Project {project_name} already exists", 400

    payload: dict[str, Any] = {
        "name": project_name,
        "path": resolved_key,
        "type": normalize_project_type(requested_type, resolved_path),
    }

    if dev_server:
        payload["dev_server"] = dev_server

    existing_names.add(project_name)
    existing_paths[resolved_key] = project_name
    return payload, None, None


def parse_excluded_dirs(raw_value: Optional[str]) -> set[str]:
    """Parse comma-separated excluded directory names."""
    excluded = set(DEFAULT_SCAN_EXCLUDE_DIRS)
    if raw_value is None:
        return excluded

    for item in raw_value.split(","):
        name = item.strip()
        if name:
            excluded.add(name)
    return excluded


def _detect_project_candidate(path: Path) -> tuple[str, str] | None:
    if (path / "pubspec.yaml").exists():
        return "flutter", "pubspec.yaml"

    package_json = path / "package.json"
    if package_json.exists():
        project_type = infer_project_type(path)
        if project_type == "other":
            project_type = "web"
        return project_type, "package.json"

    for marker in NEXT_CONFIG_MARKERS:
        if (path / marker).exists():
            return "nextjs", marker

    for marker in ("pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml"):
        if (path / marker).exists():
            return "other", marker

    if (path / ".git").is_dir():
        return "other", ".git"

    return None


def scan_project_candidates(
    root_path: Path,
    excluded_dirs: set[str],
    max_depth: int,
) -> list[dict[str, str]]:
    """Scan a root folder and collect candidate project directories."""
    visited: set[str] = set()
    candidates: dict[str, dict[str, str]] = {}
    queue: deque[tuple[Path, int]] = deque([(root_path, 0)])

    while queue and len(candidates) < MAX_SCAN_RESULTS:
        current, depth = queue.popleft()
        current_key = str(current)
        if current_key in visited:
            continue
        visited.add(current_key)

        detected = _detect_project_candidate(current)
        if detected is not None:
            project_type, marker = detected
            candidates[current_key] = {
                "name": current.name or current_key,
                "path": current_key,
                "type": project_type,
                "marker": marker,
            }

        if depth >= max_depth:
            continue

        try:
            children = sorted(
                (child for child in current.iterdir() if child.is_dir()),
                key=lambda child: child.name.lower(),
            )
        except (PermissionError, OSError):
            continue

        for child in children:
            if child.name in excluded_dirs:
                continue
            queue.append((child, depth + 1))

    return sorted(candidates.values(), key=lambda item: item["path"].lower())
