"""Project management API routes."""

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from models import DeviceRunRequest, ProjectCreate, ProjectImport, ProjectUpdate
from project_action_service import (
    build_project_flutter_web_for_current_server,
    close_project_session_for_current_server,
    create_project_record_for_current_server,
    delete_project_record_for_current_server,
    get_project_build_status_for_current_server,
    get_project_device_run_log_for_current_server,
    get_project_for_current_server,
    import_project_records_for_current_server,
    list_projects_for_current_server,
    run_project_on_device_for_current_server,
    start_project_dev_server_for_current_server,
    stop_project_dev_server_for_current_server,
    stop_project_device_run_for_current_server,
    update_project_record_for_current_server,
)
from .deps import verify_api_key
from .result_response import as_flagged_response, as_route_response

router = APIRouter(tags=["projects"])


def _project_not_found_response(name: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})


def _project_action_response(result: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    return as_flagged_response(result, error_status_code=400)


@router.get("/api/projects", dependencies=[Depends(verify_api_key)])
async def list_projects():
    """List all configured projects."""
    return {"projects": list_projects_for_current_server()}


@router.get("/api/projects/{name}", dependencies=[Depends(verify_api_key)])
async def get_project(name: str):
    """Get project details."""
    project = get_project_for_current_server(name)

    if project is None:
        return _project_not_found_response(name)

    return project


@router.post("/api/projects", dependencies=[Depends(verify_api_key)])
async def create_project(project: ProjectCreate):
    """Create a new project."""
    dev_server = project.dev_server.model_dump(exclude_none=True) if project.dev_server else None
    result = create_project_record_for_current_server(
        path_value=project.path,
        requested_name=project.name,
        requested_type=project.type,
        dev_server=dev_server,
    )
    return as_route_response(result)


@router.post("/api/projects/import", dependencies=[Depends(verify_api_key)])
async def import_projects(project_import: ProjectImport):
    """Import multiple projects by absolute paths."""
    result = import_project_records_for_current_server(project_import.paths)
    return as_route_response(result)


@router.put("/api/projects/{name}", dependencies=[Depends(verify_api_key)])
async def update_project(name: str, project: ProjectUpdate):
    """Update an existing project."""
    result = update_project_record_for_current_server(
        name,
        project.model_dump(exclude_unset=True),
    )
    return as_route_response(result)


@router.delete("/api/projects/{name}", dependencies=[Depends(verify_api_key)])
async def delete_project(name: str):
    """Delete a project."""
    result = await delete_project_record_for_current_server(name)
    return as_route_response(result)


@router.post("/api/projects/{name}/start", dependencies=[Depends(verify_api_key)])
async def start_dev_server(name: str):
    """Start dev server for project."""
    result = await start_project_dev_server_for_current_server(name)
    return _project_action_response(result)


@router.post("/api/projects/{name}/stop", dependencies=[Depends(verify_api_key)])
async def stop_dev_server(name: str):
    """Stop dev server for project."""
    result = await stop_project_dev_server_for_current_server(name)
    return _project_action_response(result)


@router.post("/api/projects/{name}/run-device", dependencies=[Depends(verify_api_key)])
async def run_project_on_device(name: str, request: DeviceRunRequest):
    """Run Flutter project on selected Android device and capture logs."""
    result = await run_project_on_device_for_current_server(name, request.device_id)
    return _project_action_response(result)


@router.post("/api/projects/{name}/stop-device-run", dependencies=[Depends(verify_api_key)])
async def stop_project_on_device(name: str):
    """Stop running Flutter device process for project."""
    result = await stop_project_device_run_for_current_server(name)
    return _project_action_response(result)


@router.get("/api/projects/{name}/device-run-log", dependencies=[Depends(verify_api_key)])
async def get_project_device_run_log(name: str, lines: int = 120):
    """Get captured log tail for the project's latest Flutter device run."""
    return get_project_device_run_log_for_current_server(name, lines=lines)


@router.post("/api/projects/{name}/build", dependencies=[Depends(verify_api_key)])
async def build_flutter_web(name: str):
    """Build Flutter web app."""
    result = await build_project_flutter_web_for_current_server(name)
    return _project_action_response(result)


@router.get("/api/projects/{name}/build-status", dependencies=[Depends(verify_api_key)])
async def get_build_status(name: str):
    """Get Flutter web build status."""
    return get_project_build_status_for_current_server(name)


@router.post("/api/sessions/{project_name}/close", dependencies=[Depends(verify_api_key)])
async def close_session(project_name: str):
    """Close Claude session for a project."""
    return await close_project_session_for_current_server(project_name)
