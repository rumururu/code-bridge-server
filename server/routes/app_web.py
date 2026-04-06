"""Flutter web app static serving routes."""

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app_web_service import (
    get_flutter_media_type_for_current_server,
    render_flutter_index_for_current_server,
    resolve_flutter_web_file_for_current_server,
)

router = APIRouter(tags=["app"])


@router.get("/app/{path:path}")
async def serve_flutter_app(path: str = ""):
    """Serve Flutter web app for iPad/Safari access."""
    resolution = resolve_flutter_web_file_for_current_server(path)
    if not resolution.success:
        return JSONResponse(
            status_code=resolution.status_code,
            content=resolution.error_payload or {"error": "Unknown error"},
        )

    file_path = resolution.file_path
    if file_path is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Resolved file path is missing"},
        )

    if file_path.name == "index.html":
        content = render_flutter_index_for_current_server(file_path)
        return HTMLResponse(content=content)

    media_type = get_flutter_media_type_for_current_server(file_path)
    return FileResponse(file_path, media_type=media_type)
