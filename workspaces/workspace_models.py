"""Pydantic models for Agent Cockpit workspaces."""

from typing import Any

from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    """Request body for creating a workspace."""

    project_name: str | None = None
    type: str = "code_project"
    root_path: str
    display_name: str | None = None
    permissions: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class WorkspaceUpdate(BaseModel):
    """Request body for updating a workspace."""

    project_name: str | None = None
    type: str | None = None
    root_path: str | None = None
    display_name: str | None = None
    permissions: dict[str, Any] | None = None
    status: str | None = None
