"""File browser and file operations API routes."""

from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from agent.tool_artifacts import record_tool_action_result
from audit.route_audit import record_api_action
from files.file_action_service import (
    copy_project_path_for_current_server,
    create_project_file_or_directory_for_current_server,
    delete_project_path_for_current_server,
    list_project_files_for_current_server,
    move_project_path_for_current_server,
    read_project_file_for_current_server,
    rename_project_path_for_current_server,
    search_project_file_content_for_current_server,
    search_project_files_for_current_server,
    upload_project_attachment_for_current_server,
    write_project_file_content_for_current_server,
)
from models import FileCreate, FileWrite
from policy.policy_gate import evaluate_direct_action_gate
from .deps import verify_api_key
from .result_response import as_route_response

router = APIRouter(tags=["files"])


def _gate_response(gate: dict[str, Any]) -> JSONResponse | None:
    if gate["allowed"]:
        return None
    return JSONResponse(status_code=int(gate["status_code"]), content=gate["payload"])


def _record_file_result(
    *,
    operation: str,
    project_name: str,
    run_id: str | None,
    details: dict[str, Any],
    result: Any,
) -> None:
    record_api_action(
        operation=operation,
        project_name=project_name,
        run_id=run_id,
        details=details,
        success=result.success,
        status_code=result.status_code,
    )
    record_tool_action_result(
        run_id=run_id,
        operation=operation,
        project_name=project_name,
        details=details,
        result=result,
    )


@router.get("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_files(name: str, path: str = "") -> dict[str, Any] | Response:
    """List directory contents for a project."""
    result = list_project_files_for_current_server(name, path)
    return as_route_response(result)


@router.get("/api/projects/{name}/files/content", dependencies=[Depends(verify_api_key)], response_model=None)
async def read_file_content(name: str, path: str) -> dict[str, Any] | Response:
    """Read file content for a project."""
    result = read_project_file_for_current_server(name, path)
    return as_route_response(result)


@router.put("/api/projects/{name}/files/content", dependencies=[Depends(verify_api_key)], response_model=None)
async def write_file_content(
    name: str,
    file_data: FileWrite,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Write content to a file."""
    details = {
        "path": file_data.path,
        "create_dirs": file_data.create_dirs,
        "content_bytes": len(file_data.content.encode("utf-8")),
    }
    gate = evaluate_direct_action_gate(
        operation="file.write",
        project_name=name,
        run_id=file_data.run_id,
        details=details,
        require_approval=require_approval,
        approval_id=file_data.approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = write_project_file_content_for_current_server(
        name,
        file_data.path,
        file_data.content,
        create_dirs=file_data.create_dirs,
    )
    _record_file_result(
        operation="file.write",
        project_name=name,
        run_id=file_data.run_id,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_file_or_directory(
    name: str,
    file_data: FileCreate,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Create a new file or directory."""
    content = file_data.content or ""
    details = {
        "path": file_data.path,
        "is_directory": file_data.is_directory,
        "content_bytes": len(content.encode("utf-8")),
    }
    gate = evaluate_direct_action_gate(
        operation="file.write",
        project_name=name,
        run_id=file_data.run_id,
        details=details,
        require_approval=require_approval,
        approval_id=file_data.approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = create_project_file_or_directory_for_current_server(
        name,
        file_data.path,
        is_directory=file_data.is_directory,
        content=content,
    )
    _record_file_result(
        operation="file.write",
        project_name=name,
        run_id=file_data.run_id,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.delete("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)], response_model=None)
async def delete_file(
    name: str,
    path: str,
    recursive: bool = False,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Delete a file or directory."""
    details = {"path": path, "recursive": recursive}
    gate = evaluate_direct_action_gate(
        operation="file.delete",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = delete_project_path_for_current_server(name, path, recursive=recursive)
    _record_file_result(
        operation="file.delete",
        project_name=name,
        run_id=run_id,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/files/rename", dependencies=[Depends(verify_api_key)], response_model=None)
async def rename_file(
    name: str,
    old_path: str,
    new_path: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Rename or move a file/directory."""
    details = {"source": old_path, "dest": new_path}
    gate = evaluate_direct_action_gate(
        operation="file.move",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = rename_project_path_for_current_server(name, old_path, new_path)
    _record_file_result(
        operation="file.move",
        project_name=name,
        run_id=run_id,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/files/copy", dependencies=[Depends(verify_api_key)], response_model=None)
async def copy_file(
    name: str,
    source: str,
    dest: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Copy a file or directory."""
    details = {"source": source, "dest": dest}
    gate = evaluate_direct_action_gate(
        operation="file.copy",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = copy_project_path_for_current_server(name, source, dest)
    _record_file_result(
        operation="file.copy",
        project_name=name,
        run_id=run_id,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/files/move", dependencies=[Depends(verify_api_key)], response_model=None)
async def move_file(
    name: str,
    source: str,
    dest: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Move a file or directory (alias for rename)."""
    details = {"source": source, "dest": dest}
    gate = evaluate_direct_action_gate(
        operation="file.move",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = move_project_path_for_current_server(name, source, dest)
    _record_file_result(
        operation="file.move",
        project_name=name,
        run_id=run_id,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.get("/api/projects/{name}/files/search", dependencies=[Depends(verify_api_key)], response_model=None)
async def search_files(name: str, q: str, limit: int = 50) -> dict[str, Any] | Response:
    """Search files in a project."""
    result = search_project_files_for_current_server(name, q, limit=limit)
    return as_route_response(result)


@router.get("/api/projects/{name}/files/search-content", dependencies=[Depends(verify_api_key)], response_model=None)
async def search_file_content(
    name: str,
    q: str,
    limit: int = 100,
    case_sensitive: bool = False,
) -> dict[str, Any] | Response:
    """Search file contents in a project."""
    result = search_project_file_content_for_current_server(
        name,
        q,
        limit=limit,
        case_sensitive=case_sensitive,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/files/upload", dependencies=[Depends(verify_api_key)], response_model=None)
async def upload_attachment(
    name: str,
    file: UploadFile = File(...),
    source: str = Form("file"),
    run_id: str | None = Form(None),
    approval_id: str | None = Form(None),
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Upload one attachment into the project workspace."""
    content = await file.read()
    details = {
        "filename": file.filename or "attachment",
        "content_type": file.content_type,
        "source": source,
        "content_bytes": len(content),
    }
    gate = evaluate_direct_action_gate(
        operation="file.upload",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = upload_project_attachment_for_current_server(
        name,
        filename=file.filename or "attachment",
        content=content,
        content_type=file.content_type,
        source=source,
    )
    _record_file_result(
        operation="file.upload",
        project_name=name,
        run_id=run_id,
        details=details,
        result=result,
    )
    return as_route_response(result)
