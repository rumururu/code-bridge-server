"""Helpers for serving the Flutter web app route."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_preview import MIME_TYPES

# Path to Flutter web build output.
FLUTTER_WEB_BUILD = Path(__file__).resolve().parents[1] / "build" / "web"


@dataclass(frozen=True)
class AppWebFileResolution:
    """Result of resolving one /app file request."""

    success: bool
    status_code: int
    file_path: Path | None = None
    error_payload: dict[str, Any] | None = None


def resolve_flutter_web_file_for_current_server(
    path: str = "",
    *,
    build_root: Path = FLUTTER_WEB_BUILD,
) -> AppWebFileResolution:
    """Resolve requested app path to a file inside Flutter web build output."""
    if not build_root.exists():
        return AppWebFileResolution(
            success=False,
            status_code=404,
            error_payload={
                "error": "Flutter web build not found",
                "hint": "Run 'flutter build web' first",
            },
        )

    file_path = build_root / (path or "index.html")

    if not file_path.exists() or file_path.is_dir():
        if file_path.is_dir():
            file_path = file_path / "index.html"
        if not file_path.exists():
            file_path = build_root / "index.html"

    if not file_path.exists():
        return AppWebFileResolution(
            success=False,
            status_code=404,
            error_payload={"error": f"File not found: {path}"},
        )

    try:
        resolved = file_path.resolve()
        resolved.relative_to(build_root.resolve())
    except ValueError:
        return AppWebFileResolution(
            success=False,
            status_code=403,
            error_payload={"error": "Access denied"},
        )

    return AppWebFileResolution(
        success=True,
        status_code=200,
        file_path=file_path,
    )


def render_flutter_index_for_current_server(file_path: Path) -> str:
    """Render index.html with /app base path."""
    content = file_path.read_text()
    return content.replace('<base href="/">', '<base href="/app/">')


def get_flutter_media_type_for_current_server(file_path: Path) -> str:
    """Get response media type for static file path."""
    suffix = file_path.suffix.lower()
    return MIME_TYPES.get(suffix, "application/octet-stream")
