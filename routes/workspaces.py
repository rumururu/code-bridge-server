"""Workspace APIs for Agent Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from workspaces.workspace_models import WorkspaceCreate, WorkspaceUpdate
from workspaces.workspace_snapshot_service import build_workspace_snapshot
from workspaces.workspace_store import get_workspace_store

from .deps import verify_api_key

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


@router.post("", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_workspace(body: WorkspaceCreate) -> dict[str, Any]:
    """Create a workspace."""
    workspace = get_workspace_store().create_workspace(
        project_name=body.project_name,
        type=body.type,
        root_path=body.root_path,
        display_name=body.display_name,
        permissions=body.permissions,
        status=body.status,
    )
    return {"workspace": workspace}


@router.get("", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_workspaces(
    project_name: str | None = None,
    status: str | None = "active",
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List workspaces."""
    return {
        "workspaces": get_workspace_store().list_workspaces(
            project_name=project_name,
            status=status,
            limit=limit,
        )
    }


@router.get("/{workspace_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_workspace(workspace_id: str) -> dict[str, Any]:
    """Get a workspace by id."""
    workspace = get_workspace_store().get_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
    return {"workspace": workspace}


@router.get("/{workspace_id}/snapshot", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_workspace_snapshot(
    workspace_id: str,
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    """Return a shallow filesystem snapshot for one workspace."""
    snapshot = build_workspace_snapshot(workspace_id, limit=limit)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
    return {"snapshot": snapshot}


@router.patch("/{workspace_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_workspace(workspace_id: str, body: WorkspaceUpdate) -> dict[str, Any]:
    """Update a workspace."""
    workspace = get_workspace_store().update_workspace(
        workspace_id,
        project_name=body.project_name,
        type=body.type,
        root_path=body.root_path,
        display_name=body.display_name,
        permissions=body.permissions,
        status=body.status,
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
    return {"workspace": workspace}


@router.post("/{workspace_id}/archive", dependencies=[Depends(verify_api_key)], response_model=None)
async def archive_workspace(workspace_id: str) -> dict[str, Any]:
    """Archive a workspace without deleting its historical runs."""
    workspace = get_workspace_store().archive_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
    return {"workspace": workspace}
