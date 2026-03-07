"""Debug routes."""

from fastapi import APIRouter, Depends

from system_status_service import get_debug_port_snapshot_for_current_server

from .deps import require_local_access

router = APIRouter(tags=["debug"])


@router.get("/api/debug/port", dependencies=[Depends(require_local_access)])
async def debug_port():
    """Debug endpoint to inspect effective port configuration."""
    return get_debug_port_snapshot_for_current_server()
