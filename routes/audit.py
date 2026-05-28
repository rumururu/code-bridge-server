"""Local audit log APIs for Agent Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, Query

from audit.audit_store import get_audit_store

from .deps import verify_api_key

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/events", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_audit_events(
    run_id: str | None = None,
    project_name: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List local audit events."""
    return {
        "events": get_audit_store().list_events(
            run_id=run_id,
            project_name=project_name,
            limit=limit,
        )
    }


@router.get("/runs/{run_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_run_audit_events(run_id: str) -> dict[str, Any]:
    """List audit events for one agent run."""
    return {"events": get_audit_store().list_events(run_id=run_id)}


@router.post("/export", dependencies=[Depends(verify_api_key)], response_model=None)
async def export_audit_events(
    run_id: str | None = None,
    project_name: str | None = None,
    limit: int = Query(default=500, ge=1, le=500),
) -> dict[str, Any]:
    """Return an in-band audit export bundle.

    File-based export will be added in the desktop Agent Host layer.
    """
    events = get_audit_store().list_events(
        run_id=run_id,
        project_name=project_name,
        limit=limit,
    )
    return {
        "format": "code_bridge.audit.v1",
        "count": len(events),
        "events": events,
    }

