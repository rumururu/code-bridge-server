"""System inspection routes: directory browse, candidate scan, and usage."""

from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from system.system_inspect_service import (
    get_system_usage_for_current_server,
    list_project_candidates_for_current_server,
    list_system_directories_for_current_server,
)
from .deps import verify_api_key
from .result_response import as_route_response

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/directories", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_system_directories(path: Optional[str] = None) -> dict[str, Any] | Response:
    """List subdirectories for an absolute server path."""
    result = list_system_directories_for_current_server(path)
    return as_route_response(result)


@router.get("/project-candidates", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_project_candidates(
    root_path: str,
    exclude_dirs: Optional[str] = None,
    max_depth: int = 1,
) -> dict[str, Any] | Response:
    """Scan a root directory and return candidate project folders."""
    result = list_project_candidates_for_current_server(
        root_path,
        exclude_dirs=exclude_dirs,
        max_depth=max_depth,
    )
    return as_route_response(result)


@router.get("/usage", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_system_usage() -> dict[str, Any] | Response:
    """Get rolling weekly usage summary and budget percentage."""
    result = await get_system_usage_for_current_server()
    return as_route_response(result)
