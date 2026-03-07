"""Planning helpers for Flutter device-run requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project_models import ProjectType


@dataclass(frozen=True)
class DeviceRunPlan:
    """Resolved input for launching flutter run on one device."""

    project_path: str
    device_id: str


@dataclass(frozen=True)
class DeviceRunPlanResult:
    """Validation result for one device run request."""

    success: bool
    plan: DeviceRunPlan | None = None
    error_message: str | None = None


def resolve_device_run_plan(
    name: str,
    device_id: str,
    project: dict[str, Any] | None,
) -> DeviceRunPlanResult:
    """Validate request and project metadata for flutter run."""
    normalized_device_id = device_id.strip()
    if not normalized_device_id:
        return DeviceRunPlanResult(success=False, error_message="Device ID is required")

    if not project:
        return DeviceRunPlanResult(success=False, error_message=f"Project {name} not found")

    project_type = ProjectType.from_string(project.get("type", ""))
    if project_type != ProjectType.FLUTTER:
        return DeviceRunPlanResult(
            success=False,
            error_message="Only Flutter projects support device run",
        )

    project_path = project.get("path")
    if not project_path or not Path(project_path).exists():
        return DeviceRunPlanResult(
            success=False,
            error_message=f"Project path does not exist: {project_path}",
        )

    return DeviceRunPlanResult(
        success=True,
        plan=DeviceRunPlan(
            project_path=str(project_path),
            device_id=normalized_device_id,
        ),
    )
