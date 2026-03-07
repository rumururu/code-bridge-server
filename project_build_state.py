"""State helpers for project web-build lifecycle."""

from __future__ import annotations

from typing import Any

from project_models import BuildInfo, BuildStatus


def mark_building(build_info: dict[str, BuildInfo], name: str, project_type: str) -> None:
    """Mark project build as currently in progress."""
    build_info[name] = BuildInfo(status=BuildStatus.BUILDING, project_type=project_type)


def mark_build_error(
    build_info: dict[str, BuildInfo],
    name: str,
    *,
    error_message: str,
    project_type: str | None = None,
) -> None:
    """Mark project build as failed."""
    build_info[name] = BuildInfo(
        status=BuildStatus.ERROR,
        error_message=error_message,
        project_type=project_type,
    )


def mark_build_ready(
    build_info: dict[str, BuildInfo],
    name: str,
    *,
    build_path: str | None,
    project_type: str,
) -> None:
    """Mark project build as ready and store build path."""
    build_info[name] = BuildInfo(
        status=BuildStatus.READY,
        build_path=build_path,
        project_type=project_type,
    )


def build_status_payload(info: BuildInfo | None) -> dict[str, Any]:
    """Convert optional build info into API payload."""
    if not info:
        return {
            "status": BuildStatus.NONE.value,
            "build_path": None,
            "error_message": None,
            "project_type": None,
        }

    return {
        "status": info.status.value,
        "build_path": info.build_path,
        "error_message": info.error_message,
        "project_type": info.project_type,
    }


def ready_build_path(info: BuildInfo | None) -> str | None:
    """Return build path only when status is READY."""
    if info and info.status == BuildStatus.READY:
        return info.build_path
    return None
