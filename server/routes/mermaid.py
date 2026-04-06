"""Mermaid diagram rendering API endpoint.

Renders Mermaid diagrams to SVG on the server side,
avoiding external CDN JavaScript loading in the app.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.mermaid_service import get_mermaid_service

router = APIRouter(tags=["mermaid"])


class MermaidRenderRequest(BaseModel):
    """Request model for Mermaid diagram rendering."""

    code: str = Field(..., description="Mermaid diagram code")
    theme: str = Field(default="default", description="Theme: default, dark, forest, neutral")


class MermaidRenderResponse(BaseModel):
    """Response model for Mermaid diagram rendering."""

    success: bool
    svg: str | None = None
    svg_base64: str | None = None
    error: str | None = None


@router.post("/api/render-mermaid", response_model=MermaidRenderResponse)
async def render_mermaid(request: MermaidRenderRequest) -> dict[str, Any]:
    """Render a Mermaid diagram to SVG.

    Returns both raw SVG and base64-encoded SVG for flexibility.
    The base64 version can be used directly in an img src attribute.
    """
    service = get_mermaid_service()

    if not service.is_available:
        raise HTTPException(
            status_code=503,
            detail="mermaid-cli not installed. Run: npm install -g @mermaid-js/mermaid-cli",
        )

    svg_content, error = await service.render_svg(request.code, request.theme)

    if error:
        return {
            "success": False,
            "svg": None,
            "svg_base64": None,
            "error": error,
        }

    # Also generate base64 version
    svg_base64, _ = await service.render_svg_base64(request.code, request.theme)

    return {
        "success": True,
        "svg": svg_content,
        "svg_base64": svg_base64,
        "error": None,
    }


@router.get("/api/mermaid/status")
async def mermaid_status() -> dict[str, Any]:
    """Check if mermaid-cli is available."""
    service = get_mermaid_service()
    return {
        "available": service.is_available,
        "message": "mermaid-cli is available" if service.is_available else "mermaid-cli not installed",
    }
