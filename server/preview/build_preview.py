"""Static build preview serving helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# MIME type mapping for common web files.
MIME_TYPES = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".eot": "application/vnd.ms-fontobject",
    ".wasm": "application/wasm",
}


def _resolve_preview_file_path(build_dir: Path, path: str, project_type: str) -> Path:
    file_path = build_dir / (path or "index.html")

    if not file_path.exists() or file_path.is_dir():
        if file_path.is_dir():
            file_path = file_path / "index.html"
        if not file_path.exists():
            file_path = build_dir / "index.html"

    if project_type == "nextjs" and not file_path.exists():
        if path.startswith("_next/"):
            file_path = build_dir / path
        elif not path.startswith("_next"):
            public_file = build_dir.parent / "public" / path
            if public_file.exists():
                file_path = public_file

    return file_path


def _resolve_allowed_base(build_dir: Path, project_type: str) -> Path:
    return build_dir.resolve().parent if project_type == "nextjs" else build_dir.resolve()


def _is_within_allowed_base(file_path: Path, allowed_base: Path) -> bool:
    try:
        file_path.resolve().relative_to(allowed_base)
        return True
    except ValueError:
        return False


def _render_preview_index_html(file_path: Path, project_name: str, project_type: str) -> str:
    content = file_path.read_text()
    correct_base = f"/build-preview/{project_name}/"

    if project_type == "flutter":
        content = content.replace('<base href="/">', f'<base href="{correct_base}">')

    return content


def serve_build_preview(
    project_name: str,
    path: str,
    project_manager: Any,
):
    """Serve static files from a built project output."""
    build_path = project_manager.get_build_path(project_name)
    build_status = project_manager.get_build_status(project_name)
    project_type = build_status.get("project_type", "flutter")

    if not build_path:
        return JSONResponse(
            status_code=404,
            content={"error": f"No build available for project {project_name}. Run build first."},
        )

    build_dir = Path(build_path)
    if not build_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Build directory not found: {build_path}"},
        )

    file_path = _resolve_preview_file_path(build_dir, path, project_type)

    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {path}"},
        )

    allowed_base = _resolve_allowed_base(build_dir, project_type)
    if not _is_within_allowed_base(file_path, allowed_base):
        return JSONResponse(
            status_code=403,
            content={"error": "Access denied"},
        )

    suffix = file_path.suffix.lower()
    media_type = MIME_TYPES.get(suffix, "application/octet-stream")

    if file_path.name == "index.html":
        content = _render_preview_index_html(file_path, project_name, project_type)
        return HTMLResponse(content=content)

    return FileResponse(file_path, media_type=media_type)
