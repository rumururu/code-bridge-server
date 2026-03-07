"""Preview API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from preview_route_service import (
    authorize_project_preview_request_for_current_server,
    create_preview_token_for_current_server,
    get_allowed_root_preview_proxy_path,
    proxy_preview_request_for_current_server,
    resolve_last_preview_project_target_for_current_server,
    resolve_project_preview_target_for_current_server,
    serve_build_preview_for_current_server,
    set_last_previewed_project_for_current_server,
)
from .deps import verify_api_key

router = APIRouter(tags=["preview"])


@router.post("/api/preview/token")
async def create_preview_token(
    project_name: str = Query(..., alias="project"),
    _: str = Depends(verify_api_key),
):
    """Generate a preview token for accessing dev server preview."""
    result = create_preview_token_for_current_server(project_name)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.payload.get("error"))
    return result.payload


@router.api_route("/preview/{name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def preview_proxy(
    request: Request,
    name: str,
    path: str = "",
):
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


async def _proxy_from_last_preview_project(request: Request, path: str):
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


@router.api_route("/_next/{path:path}", methods=["GET", "POST"])
async def next_assets_proxy(request: Request, path: str = ""):
    """Proxy Next.js asset requests to the active dev server."""
    return await _proxy_from_last_preview_project(request, f"_next/{path}")


@router.api_route("/@{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_at_assets_proxy(request: Request, path: str = ""):
    """Proxy root-level @* assets used by Vite and similar dev servers."""
    return await _proxy_from_last_preview_project(request, f"@{path}")


@router.api_route("/src/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_src_assets_proxy(request: Request, path: str = ""):
    """Proxy root-level src assets for dev servers."""
    return await _proxy_from_last_preview_project(request, f"src/{path}")


@router.api_route("/node_modules/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_node_modules_proxy(request: Request, path: str = ""):
    """Proxy root-level node_modules assets for dev servers."""
    return await _proxy_from_last_preview_project(request, f"node_modules/{path}")


@router.api_route("/assets/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_assets_proxy(request: Request, path: str = ""):
    """Proxy root-level assets folder for dev servers."""
    return await _proxy_from_last_preview_project(request, f"assets/{path}")


@router.api_route("/{filename}", methods=["GET"])
async def root_file_proxy(request: Request, filename: str):
    """Proxy common root-level files requested by dev server HTML."""
    proxy_path = get_allowed_root_preview_proxy_path(filename)
    if proxy_path is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return await _proxy_from_last_preview_project(request, proxy_path)


@router.get("/build-preview/{name}/{path:path}")
async def build_preview(name: str, path: str = ""):
    """Serve static files from web build (Flutter or Next.js)."""
    return serve_build_preview_for_current_server(name, path)
