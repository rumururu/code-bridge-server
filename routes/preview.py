"""Preview API routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from agent.tool_artifacts import record_tool_action_result
from audit.route_audit import record_api_action
from core.base_result import BaseRouteResult
from preview.preview_route_service import (
    authorize_project_preview_request_for_current_server,
    create_preview_token_for_current_server,
    get_allowed_root_preview_proxy_path,
    proxy_preview_request_for_current_server,
    resolve_last_preview_project_target_for_current_server,
    resolve_project_preview_target_for_current_server,
    set_last_previewed_project_for_current_server,
)
from .deps import verify_api_key

router = APIRouter(tags=["preview"])


@router.post("/api/preview/token")
async def create_preview_token(
    project_name: str = Query(..., alias="project"),
    run_id: str | None = None,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Generate a preview token for accessing dev server preview."""
    result = create_preview_token_for_current_server(project_name, api_key=api_key)
    details = {"project_name": project_name}
    safe_payload = {
        "success": result.success,
        "project": project_name,
        "expires_in_minutes": result.payload.get("expires_in_minutes"),
        "preview_path": f"/preview/{project_name}/",
    }
    record_api_action(
        operation="preview.token",
        project_name=project_name,
        run_id=run_id,
        details=details,
        success=result.success,
        status_code=result.status_code,
    )
    record_tool_action_result(
        run_id=run_id,
        operation="preview.token",
        project_name=project_name,
        details=details,
        result=BaseRouteResult(
            success=result.success,
            status_code=result.status_code,
            payload=safe_payload if result.success else result.payload,
        ),
    )
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.payload.get("error"))
    return result.payload


@router.api_route("/preview/{name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], response_model=None)
async def preview_proxy(
    request: Request,
    name: str,
    path: str = "",
) -> Response | JSONResponse:
    """Proxy requests to dev server with preview token authentication."""
    auth_result = authorize_project_preview_request_for_current_server(request, name)
    if not auth_result.success:
        return JSONResponse(
            status_code=auth_result.status_code,
            content=auth_result.payload,
        )

    target_result = resolve_project_preview_target_for_current_server(name)
    if not target_result.success:
        return JSONResponse(
            status_code=target_result.status_code,
            content=target_result.payload,
        )

    set_last_previewed_project_for_current_server(name)
    return await proxy_preview_request_for_current_server(
        request,
        target_result.payload["port"],
        path,
    )


async def _proxy_from_last_preview_project(request: Request, path: str) -> Response | JSONResponse:
    """Proxy root-level asset requests for the most recently previewed project."""
    target_result = resolve_last_preview_project_target_for_current_server(request)
    if not target_result.success:
        return JSONResponse(
            status_code=target_result.status_code,
            content=target_result.payload,
        )

    return await proxy_preview_request_for_current_server(
        request,
        target_result.payload["port"],
        path,
    )


@router.api_route("/_next/{path:path}", methods=["GET", "POST"], response_model=None)
async def next_assets_proxy(request: Request, path: str = "") -> Response | JSONResponse:
    """Proxy Next.js asset requests to the active dev server."""
    return await _proxy_from_last_preview_project(request, f"_next/{path}")


@router.api_route("/@{path:path}", methods=["GET", "POST", "PUT", "DELETE"], response_model=None)
async def root_at_assets_proxy(request: Request, path: str = "") -> Response | JSONResponse:
    """Proxy root-level @* assets used by Vite and similar dev servers."""
    return await _proxy_from_last_preview_project(request, f"@{path}")


@router.api_route("/src/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], response_model=None)
async def root_src_assets_proxy(request: Request, path: str = "") -> Response | JSONResponse:
    """Proxy root-level src assets for dev servers."""
    return await _proxy_from_last_preview_project(request, f"src/{path}")


@router.api_route("/node_modules/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], response_model=None)
async def root_node_modules_proxy(request: Request, path: str = "") -> Response | JSONResponse:
    """Proxy root-level node_modules assets for dev servers."""
    return await _proxy_from_last_preview_project(request, f"node_modules/{path}")


@router.api_route("/assets/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], response_model=None)
async def root_assets_proxy(request: Request, path: str = "") -> Response | JSONResponse:
    """Proxy root-level assets folder for dev servers."""
    return await _proxy_from_last_preview_project(request, f"assets/{path}")


@router.api_route("/{filename}", methods=["GET"], response_model=None)
async def root_file_proxy(request: Request, filename: str) -> Response | JSONResponse:
    """Proxy common root-level files requested by dev server HTML."""
    proxy_path = get_allowed_root_preview_proxy_path(filename)
    if proxy_path is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return await _proxy_from_last_preview_project(request, proxy_path)
