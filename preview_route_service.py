"""Wrappers for preview route actions and authorization checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from build_preview import serve_build_preview
from preview import get_preview_proxy
from preview_access import get_preview_access_manager
from projects import get_project_manager

ALLOWED_ROOT_PREVIEW_FILES = {
    "favicon.ico",
    "vite.svg",
    "manifest.json",
    "robots.txt",
    "sitemap.xml",
}


@dataclass(frozen=True)
class PreviewRouteResult:
    """Typed result for preview route responses."""

    success: bool
    status_code: int
    payload: dict[str, Any]

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize for route response helpers."""
        return self.payload


def _error(status_code: int, message: str, **extra: Any) -> PreviewRouteResult:
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return PreviewRouteResult(success=False, status_code=status_code, payload=payload)


def create_preview_token_for_current_server(
    project_name: str,
    *,
    manager: Any | None = None,
    preview_access: Any | None = None,
) -> PreviewRouteResult:
    """Create preview token for a running project's dev server."""
    resolved_manager = manager or get_project_manager()
    resolved_preview_access = preview_access or get_preview_access_manager()

    port = resolved_manager.get_server_port(project_name)
    if port is None:
        return _error(404, f"No running dev server for project {project_name}")

    token = resolved_preview_access.generate_preview_token(project_name)
    return PreviewRouteResult(
        success=True,
        status_code=200,
        payload={
            "token": token,
            "project": project_name,
            "expires_in_minutes": resolved_preview_access.ttl_minutes,
            "preview_url": f"/preview/{project_name}/?preview_token={token}",
        },
    )


def authorize_project_preview_request_for_current_server(
    request: Request,
    project_name: str,
    *,
    preview_access: Any | None = None,
) -> PreviewRouteResult:
    """Validate preview authorization for direct /preview/{project} requests."""
    resolved_preview_access = preview_access or get_preview_access_manager()
    is_local = resolved_preview_access.is_local_request(request)

    if is_local:
        return PreviewRouteResult(
            success=True,
            status_code=200,
            payload={"project": project_name, "is_local": True},
        )

    preview_token = request.query_params.get("preview_token")
    if preview_token:
        if not resolved_preview_access.validate_preview_token(preview_token, project_name):
            return _error(403, "Invalid or expired preview token")
        resolved_preview_access.bind_remote_session(request, project_name)
        return PreviewRouteResult(
            success=True,
            status_code=200,
            payload={"project": project_name, "is_local": False},
        )

    if not resolved_preview_access.has_remote_session(request, project_name):
        return _error(
            403,
            "Preview token required for remote access",
            hint="Call POST /api/preview/token?project=NAME to get a token",
        )

    return PreviewRouteResult(
        success=True,
        status_code=200,
        payload={"project": project_name, "is_local": False},
    )


def resolve_project_preview_target_for_current_server(
    project_name: str,
    *,
    manager: Any | None = None,
) -> PreviewRouteResult:
    """Resolve running dev-server port for one project."""
    resolved_manager = manager or get_project_manager()
    port = resolved_manager.get_server_port(project_name)

    if port is None:
        return _error(404, f"No running dev server found for project {project_name}")

    return PreviewRouteResult(
        success=True,
        status_code=200,
        payload={"project": project_name, "port": port},
    )


def set_last_previewed_project_for_current_server(
    project_name: str,
    *,
    preview_access: Any | None = None,
) -> None:
    """Persist most recently previewed project name."""
    resolved_preview_access = preview_access or get_preview_access_manager()
    resolved_preview_access.set_last_previewed_project(project_name)


def resolve_last_preview_project_target_for_current_server(
    request: Request,
    *,
    manager: Any | None = None,
    preview_access: Any | None = None,
) -> PreviewRouteResult:
    """Resolve last-preview project port for root asset proxy routes."""
    resolved_preview_access = preview_access or get_preview_access_manager()
    project_name = resolved_preview_access.get_last_previewed_project()
    if not project_name:
        return _error(404, "No active preview session")

    is_local = resolved_preview_access.is_local_request(request)
    if not is_local and not resolved_preview_access.has_remote_session(request, project_name):
        return _error(403, "Preview session not authorized")

    resolved_manager = manager or get_project_manager()
    port = resolved_manager.get_server_port(project_name)
    if port is None:
        return _error(404, f"No dev server running for {project_name}")

    return PreviewRouteResult(
        success=True,
        status_code=200,
        payload={"project": project_name, "port": port},
    )


async def proxy_preview_request_for_current_server(
    request: Request,
    target_port: int,
    path: str = "",
    *,
    proxy: Any | None = None,
):
    """Proxy a request to the resolved dev server."""
    resolved_proxy = proxy or get_preview_proxy()
    return await resolved_proxy.proxy_request(request, target_port, path)


def get_allowed_root_preview_proxy_path(filename: str) -> str | None:
    """Return allowed root proxy path or None when blocked."""
    if "/" in filename:
        return None
    if filename not in ALLOWED_ROOT_PREVIEW_FILES:
        return None
    return filename


def serve_build_preview_for_current_server(
    project_name: str,
    path: str,
    *,
    manager: Any | None = None,
):
    """Serve static files from built output for project."""
    resolved_manager = manager or get_project_manager()
    return serve_build_preview(project_name, path, resolved_manager)
