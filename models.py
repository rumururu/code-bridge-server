"""Pydantic models for Code Bridge API."""

from typing import Optional

from pydantic import BaseModel


class DevServerConfig(BaseModel):
    """Dev server configuration."""

    command: Optional[str] = None
    port: Optional[int] = None


class ProjectCreate(BaseModel):
    """Request body for creating a project."""

    path: str
    name: Optional[str] = None
    type: Optional[str] = None
    dev_server: Optional[DevServerConfig] = None


class ProjectUpdate(BaseModel):
    """Request body for updating a project."""

    path: Optional[str] = None
    type: Optional[str] = None
    dev_server: Optional[DevServerConfig] = None


class ProjectImport(BaseModel):
    """Request body for importing multiple projects by paths."""

    paths: list[str]


class DeviceRunRequest(BaseModel):
    """Request body for running a Flutter project on a connected device."""

    device_id: str


class ProjectResponse(BaseModel):
    """Response body for a project."""

    name: str
    path: str
    type: str
    dev_server: Optional[DevServerConfig] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class FileWrite(BaseModel):
    """Request body for writing a file."""

    path: str
    content: str
    create_dirs: bool = False


class FileCreate(BaseModel):
    """Request body for creating a file or directory."""

    path: str
    content: Optional[str] = None  # None for directory creation
    is_directory: bool = False


class TerminalCommand(BaseModel):
    """Request body for executing a terminal command."""

    command: str
    timeout: int = 300  # Default 5 minutes


class GitCommit(BaseModel):
    """Request body for git commit."""

    message: str


class GitBranch(BaseModel):
    """Request body for git branch operations."""

    name: str
    start_point: Optional[str] = None


class GitPush(BaseModel):
    """Request body for git push."""

    remote: str = "origin"
    branch: Optional[str] = None


class GitPull(BaseModel):
    """Request body for git pull."""

    remote: str = "origin"
    branch: Optional[str] = None


class LlmSelectionUpdate(BaseModel):
    """Request body for selecting active LLM company/model."""

    company_id: str
    model: str


class LlmCustomModelUpdate(BaseModel):
    """Request body for adding one custom model in settings."""

    company_id: str
    model: str


class PairVerifyRequest(BaseModel):
    """Request body for verifying a pairing token."""

    pair_token: str
    client_id: Optional[str] = None
    device_name: Optional[str] = None
