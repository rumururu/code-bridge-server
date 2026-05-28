"""Project management API routes."""

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from agent.tool_artifacts import record_tool_action_result
from agent.agent_store import get_agent_store
from audit.route_audit import record_api_action
from core.base_result import BaseRouteResult
from models import (
    DeviceRunRequest,
    ProjectCreate,
    ProjectFolderCreate,
    ProjectImport,
    ProjectUpdate,
    WebPreviewLaunchRequest,
)
from projects.project_action_service import (
    close_project_session_for_current_server,
    create_project_folder_for_current_server,
    create_project_record_for_current_server,
    delete_project_record_for_current_server,
    get_project_device_run_log_for_current_server,
    get_project_for_current_server,
    import_project_records_for_current_server,
    open_web_preview_on_device_for_current_server,
    list_projects_for_current_server,
    restart_project_dev_server_for_current_server,
    run_project_on_device_for_current_server,
    start_project_dev_server_for_current_server,
    stop_project_dev_server_for_current_server,
    stop_project_device_run_for_current_server,
    update_project_record_for_current_server,
)
from projects.visual_regression import build_screenshot_visual_summary
from policy.policy_gate import evaluate_direct_action_gate
from .deps import verify_api_key
from .result_response import as_flagged_response, as_route_response

router = APIRouter(tags=["projects"])


def _project_not_found_response(name: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})


def _project_action_response(result: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    return as_flagged_response(result, error_status_code=400)


def _gate_response(gate: dict[str, Any]) -> JSONResponse | None:
    if gate["allowed"]:
        return None
    return JSONResponse(status_code=int(gate["status_code"]), content=gate["payload"])


def _record_project_action_result(
    *,
    operation: str,
    project_name: str,
    run_id: str | None,
    details: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    success = bool(payload.get("success"))
    status_code = 200 if success else 400
    record_api_action(
        operation=operation,
        project_name=project_name,
        run_id=run_id,
        details=details,
        success=success,
        status_code=status_code,
    )
    record_tool_action_result(
        run_id=run_id,
        operation=operation,
        project_name=project_name,
        details=details,
        result=BaseRouteResult(
            success=success,
            status_code=status_code,
            payload=payload,
        ),
    )
    screenshot_path = payload.get("screenshot_path")
    if run_id and isinstance(screenshot_path, str) and screenshot_path:
        store = get_agent_store()
        baseline_path = _latest_screenshot_path(
            store.list_artifacts(run_id),
            project_name=project_name,
            current_path=screenshot_path,
        )
        visual_summary = build_screenshot_visual_summary(
            screenshot_path,
            baseline_path=baseline_path,
        )
        screenshot_artifact = store.add_artifact(
            run_id=run_id,
            kind="device_screenshot",
            path=screenshot_path,
            mime_type="image/png",
            metadata={
                "operation": operation,
                "project_name": project_name,
                "details": details,
                "visual_regression": visual_summary,
            },
        )
        visual_metadata = {
            **visual_summary,
            "operation": operation,
            "project_name": project_name,
            "details": details,
            "screenshot_artifact_id": (
                screenshot_artifact.get("id")
                if screenshot_artifact is not None
                else None
            ),
        }
        store.add_artifact(
            run_id=run_id,
            kind="visual_regression",
            path=screenshot_path,
            mime_type="application/json",
            metadata=visual_metadata,
        )
    screenshot_error = payload.get("screenshot_error")
    if run_id and isinstance(screenshot_error, str) and screenshot_error:
        get_agent_store().add_artifact(
            run_id=run_id,
            kind="device_screenshot_error",
            mime_type="application/json",
            metadata={
                "operation": operation,
                "project_name": project_name,
                "details": details,
                "error": screenshot_error,
            },
        )


def _latest_screenshot_path(
    artifacts: list[dict[str, Any]],
    *,
    project_name: str,
    current_path: str,
) -> str | None:
    for artifact in reversed(artifacts):
        if artifact.get("kind") != "device_screenshot":
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path or path == current_path:
            continue
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict) and metadata.get("project_name") not in {None, project_name}:
            continue
        return path
    return None


@router.get("/api/projects", dependencies=[Depends(verify_api_key)])
async def list_projects() -> dict[str, Any]:
    """List all configured projects."""
    return {"projects": list_projects_for_current_server()}


@router.get("/api/projects/{name}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_project(name: str) -> dict[str, Any] | JSONResponse:
    """Get project details."""
    project = get_project_for_current_server(name)

    if project is None:
        return _project_not_found_response(name)

    return project


@router.post("/api/projects", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_project(project: ProjectCreate) -> dict[str, Any] | JSONResponse:
    """Create a new project."""
    dev_server = project.dev_server.model_dump(exclude_none=True) if project.dev_server else None
    result = create_project_record_for_current_server(
        path_value=project.path,
        requested_name=project.name,
        requested_type=project.type,
        dev_server=dev_server,
    )
    return as_route_response(result)


@router.post("/api/projects/folder", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_project_folder(project: ProjectFolderCreate) -> dict[str, Any] | JSONResponse:
    """Create a new folder under a root path and register it as a project."""
    dev_server = project.dev_server.model_dump(exclude_none=True) if project.dev_server else None
    result = create_project_folder_for_current_server(
        root_path=project.root_path,
        folder_name=project.folder_name,
        requested_name=project.name,
        requested_type=project.type,
        dev_server=dev_server,
    )
    return as_route_response(result)


@router.post("/api/projects/import", dependencies=[Depends(verify_api_key)], response_model=None)
async def import_projects(project_import: ProjectImport) -> dict[str, Any] | JSONResponse:
    """Import multiple projects by absolute paths."""
    result = import_project_records_for_current_server(project_import.paths)
    return as_route_response(result)


@router.put("/api/projects/{name}", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_project(name: str, project: ProjectUpdate) -> dict[str, Any] | JSONResponse:
    """Update an existing project."""
    result = update_project_record_for_current_server(
        name,
        project.model_dump(exclude_unset=True),
    )
    return as_route_response(result)


@router.delete("/api/projects/{name}", dependencies=[Depends(verify_api_key)], response_model=None)
async def delete_project(name: str) -> dict[str, Any] | JSONResponse:
    """Delete a project."""
    result = await delete_project_record_for_current_server(name)
    return as_route_response(result)


@router.post("/api/projects/{name}/start", dependencies=[Depends(verify_api_key)], response_model=None)
async def start_dev_server(
    name: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | JSONResponse:
    """Start dev server for project."""
    details = {"action": "start"}
    gate = evaluate_direct_action_gate(
        operation="process.devserver",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await start_project_dev_server_for_current_server(name)
    _record_project_action_result(
        operation="process.devserver",
        project_name=name,
        run_id=run_id,
        details=details,
        payload=result,
    )
    return _project_action_response(result)


@router.post("/api/projects/{name}/stop", dependencies=[Depends(verify_api_key)], response_model=None)
async def stop_dev_server(
    name: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | JSONResponse:
    """Stop dev server for project."""
    details = {"action": "stop"}
    gate = evaluate_direct_action_gate(
        operation="process.devserver",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await stop_project_dev_server_for_current_server(name)
    _record_project_action_result(
        operation="process.devserver",
        project_name=name,
        run_id=run_id,
        details=details,
        payload=result,
    )
    return _project_action_response(result)


@router.post("/api/projects/{name}/restart", dependencies=[Depends(verify_api_key)], response_model=None)
async def restart_dev_server(
    name: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | JSONResponse:
    """Restart dev server for project (stop then start)."""
    details = {"action": "restart"}
    gate = evaluate_direct_action_gate(
        operation="process.devserver",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await restart_project_dev_server_for_current_server(name)
    _record_project_action_result(
        operation="process.devserver",
        project_name=name,
        run_id=run_id,
        details=details,
        payload=result,
    )
    return _project_action_response(result)


@router.post("/api/projects/{name}/run-device", dependencies=[Depends(verify_api_key)], response_model=None)
async def run_project_on_device(
    name: str,
    request: DeviceRunRequest,
    require_approval: bool = False,
) -> dict[str, Any] | JSONResponse:
    """Run Flutter project on selected Android device and capture logs."""
    details = {"action": "run_device", "device_id": request.device_id}
    gate = evaluate_direct_action_gate(
        operation="device.control",
        project_name=name,
        run_id=request.run_id,
        details=details,
        require_approval=require_approval,
        approval_id=request.approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await run_project_on_device_for_current_server(name, request.device_id)
    _record_project_action_result(
        operation="device.control",
        project_name=name,
        run_id=request.run_id,
        details=details,
        payload=result,
    )
    return _project_action_response(result)


@router.post("/api/projects/{name}/open-web-preview", dependencies=[Depends(verify_api_key)], response_model=None)
async def open_web_preview_on_device(
    name: str,
    request: WebPreviewLaunchRequest,
    require_approval: bool = False,
) -> dict[str, Any] | JSONResponse:
    """Open web project preview on selected Android emulator."""
    details = {
        "action": "open_web_preview",
        "device_id": request.device_id,
        "width": request.width,
        "height": request.height,
        "density": request.density,
        "reset_to_default": request.reset_to_default,
    }
    gate = evaluate_direct_action_gate(
        operation="device.control",
        project_name=name,
        run_id=request.run_id,
        details=details,
        require_approval=require_approval,
        approval_id=request.approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await open_web_preview_on_device_for_current_server(
        name,
        request.device_id,
        width=request.width,
        height=request.height,
        density=request.density,
        reset_to_default=request.reset_to_default,
    )
    _record_project_action_result(
        operation="device.control",
        project_name=name,
        run_id=request.run_id,
        details=details,
        payload=result,
    )
    return _project_action_response(result)


@router.post("/api/projects/{name}/stop-device-run", dependencies=[Depends(verify_api_key)], response_model=None)
async def stop_project_on_device(
    name: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | JSONResponse:
    """Stop running Flutter device process for project."""
    details = {"action": "stop_device_run"}
    gate = evaluate_direct_action_gate(
        operation="device.control",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await stop_project_device_run_for_current_server(name)
    _record_project_action_result(
        operation="device.control",
        project_name=name,
        run_id=run_id,
        details=details,
        payload=result,
    )
    return _project_action_response(result)


@router.get("/api/projects/{name}/device-run-log", dependencies=[Depends(verify_api_key)])
async def get_project_device_run_log(name: str, lines: int = 120) -> dict[str, Any]:
    """Get captured log tail for the project's latest Flutter device run."""
    return get_project_device_run_log_for_current_server(name, lines=lines)


@router.post("/api/sessions/{project_name}/close", dependencies=[Depends(verify_api_key)])
async def close_session(project_name: str) -> dict[str, Any]:
    """Close Claude session for a project."""
    return await close_project_session_for_current_server(project_name)
