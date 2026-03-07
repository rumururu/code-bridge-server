"""Health check endpoint."""

from fastapi import APIRouter

from system_status_service import get_health_status_for_current_server

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return get_health_status_for_current_server()
