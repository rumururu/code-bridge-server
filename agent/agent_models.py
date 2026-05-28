"""Pydantic models for Agent Cockpit APIs."""

from typing import Any

from pydantic import BaseModel, Field


class AgentRunCreate(BaseModel):
    """Request body for creating a durable agent run."""

    project_name: str | None = None
    workspace_id: str | None = None
    provider_id: str | None = None
    model: str | None = None
    title: str | None = None
    goal: str | None = None
    cwd: str | None = None
    parent_run_id: str | None = None
    native_session_id: str | None = None
    task_id: str | None = None


class AgentRunMessageCreate(BaseModel):
    """Request body for adding a message to an agent run."""

    role: str = Field(pattern="^(user|assistant|system|tool)$")
    content: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class AgentEventCreate(BaseModel):
    """Request body for appending a normalized event to an agent run."""

    event_type: str
    provider_id: str | None = None
    provider_event: dict[str, Any] | None = None
    app_event: dict[str, Any] | None = None


class AgentArtifactCreate(BaseModel):
    """Request body for registering an artifact produced by an agent run."""

    kind: str
    path: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskCreate(BaseModel):
    """Request body for creating a tracked agent task."""

    title: str
    description: str | None = None
    project_name: str | None = None
    workspace_id: str | None = None
    run_id: str | None = None
    kind: str = "general"
    source: str = "manual"
    goal: str | None = None
    priority: int = 0
    due_at: str | None = None
    labels: list[str] = Field(default_factory=list)
    assignee: str | None = None
    requester: str | None = None
    acceptance: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskUpdate(BaseModel):
    """Request body for updating a tracked work task."""

    title: str | None = None
    description: str | None = None
    project_name: str | None = None
    workspace_id: str | None = None
    run_id: str | None = None
    kind: str | None = None
    source: str | None = None
    goal: str | None = None
    status: str | None = None
    priority: int | None = None
    due_at: str | None = None
    labels: list[str] | None = None
    assignee: str | None = None
    requester: str | None = None
    acceptance: list[str] | None = None
    metadata: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AgentTaskRunLinkCreate(BaseModel):
    """Request body for linking a run to a work task."""

    run_id: str
    role: str = "primary"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskStepUpdate(BaseModel):
    """Request body for updating an orchestrated task step."""

    title: str | None = None
    status: str | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    approval_id: str | None = None
    artifact_id: str | None = None


class AgentTaskStepRunCreate(BaseModel):
    """Request body for running one orchestrated task step."""

    input: dict[str, Any] | None = None
    approval_id: str | None = None
    require_approval: bool = False


class AgentConnectorRequestUpdate(BaseModel):
    """Request body for reviewing or completing connector requests."""

    status: str | None = None
    parameters: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AgentTaskStartCreate(BaseModel):
    """Request body for starting an orchestrated work task."""

    provider_id: str | None = None
    model: str | None = None
    cwd: str | None = None
    auto_start: bool = True
    dry_run: bool = False
    prompt: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    approval_id: str | None = None


class AgentRunPreflightCreate(BaseModel):
    """Request body for running verification commands for an agent run."""

    project_name: str | None = None
    commands: list[str] = Field(min_length=1, max_length=10)
    timeout: int = Field(default=300, ge=1, le=1800)
    approval_id: str | None = None
