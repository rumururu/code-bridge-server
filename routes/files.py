"""File browser and file operations API routes."""

from fastapi import APIRouter, Depends, File, Form, UploadFile

from file_action_service import (
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
from .deps import verify_api_key
from .result_response import as_route_response

router = APIRouter(tags=["files"])


@router.get("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)])
async def list_files(name: str, path: str = ""):
    """List directory contents for a project."""
    result = list_project_files_for_current_server(name, path)
    return as_route_response(result)


@router.get("/api/projects/{name}/files/content", dependencies=[Depends(verify_api_key)])
async def read_file_content(name: str, path: str):
    """Read file content for a project."""
    result = read_project_file_for_current_server(name, path)
    return as_route_response(result)


@router.put("/api/projects/{name}/files/content", dependencies=[Depends(verify_api_key)])
async def write_file_content(name: str, file_data: FileWrite):
    """Write content to a file."""
    result = write_project_file_content_for_current_server(
        name,
        file_data.path,
        file_data.content,
        create_dirs=file_data.create_dirs,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)])
async def create_file_or_directory(name: str, file_data: FileCreate):
    """Create a new file or directory."""
    result = create_project_file_or_directory_for_current_server(
        name,
        file_data.path,
        is_directory=file_data.is_directory,
        content=file_data.content or "",
    )
    return as_route_response(result)


@router.delete("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)])
async def delete_file(name: str, path: str, recursive: bool = False):
    """Delete a file or directory."""
    result = delete_project_path_for_current_server(name, path, recursive=recursive)
    return as_route_response(result)


@router.post("/api/projects/{name}/files/rename", dependencies=[Depends(verify_api_key)])
async def rename_file(name: str, old_path: str, new_path: str):
    """Rename or move a file/directory."""
    result = rename_project_path_for_current_server(name, old_path, new_path)
    return as_route_response(result)


@router.post("/api/projects/{name}/files/copy", dependencies=[Depends(verify_api_key)])
async def copy_file(name: str, source: str, dest: str):
    """Copy a file or directory."""
    result = copy_project_path_for_current_server(name, source, dest)
    return as_route_response(result)


@router.post("/api/projects/{name}/files/move", dependencies=[Depends(verify_api_key)])
async def move_file(name: str, source: str, dest: str):
    """Move a file or directory (alias for rename)."""
    result = move_project_path_for_current_server(name, source, dest)
    return as_route_response(result)


@router.get("/api/projects/{name}/files/search", dependencies=[Depends(verify_api_key)])
async def search_files(name: str, q: str, limit: int = 50):
    """Search files in a project."""
    result = search_project_files_for_current_server(name, q, limit=limit)
    return as_route_response(result)


@router.get("/api/projects/{name}/files/search-content", dependencies=[Depends(verify_api_key)])
async def search_file_content(
    name: str,
    q: str,
    limit: int = 100,
    case_sensitive: bool = False,
):
    """Search file contents in a project."""
    result = search_project_file_content_for_current_server(
        name,
        q,
        limit=limit,
        case_sensitive=case_sensitive,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/files/upload", dependencies=[Depends(verify_api_key)])
async def upload_attachment(
    name: str,
    file: UploadFile = File(...),
    source: str = Form("file"),
):
    """Upload one attachment into the project workspace."""
    content = await file.read()
    result = upload_project_attachment_for_current_server(
        name,
        filename=file.filename or "attachment",
        content=content,
        content_type=file.content_type,
        source=source,
    )
    return as_route_response(result)
