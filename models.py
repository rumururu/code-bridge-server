"""Pydantic models for Code Bridge API."""

from typing import Optional

from pydantic import BaseModel, Field


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


class ProjectFolderCreate(BaseModel):
    """Request body for creating a project folder under a root path."""

    root_path: str
    folder_name: str
    name: Optional[str] = None
    type: Optional[str] = None
    dev_server: Optional[DevServerConfig] = None


class ProjectUpdate(BaseModel):
    """Request body for updating a project."""

    path: Optional[str] = None
    type: Optional[str] = None
    dev_server: Optional[DevServerConfig] = None
    enabled: Optional[bool] = None


class ProjectImport(BaseModel):
    """Request body for importing multiple projects by paths."""

    paths: list[str]


class DeviceRunRequest(BaseModel):
    """Request body for running a Flutter project on a connected device."""

    device_id: str
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


class WebPreviewLaunchRequest(BaseModel):
    """Request body for opening a web project's dev server on an Android emulator."""

    device_id: str
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    density: Optional[int] = Field(default=None, gt=0)
    reset_to_default: bool = False
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


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
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


class FileCreate(BaseModel):
    """Request body for creating a file or directory."""

    path: str
    content: Optional[str] = None  # None for directory creation
    is_directory: bool = False
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


class TerminalCommand(BaseModel):
    """Request body for executing a terminal command."""

    command: str
    timeout: int = 300  # Default 5 minutes
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


class GitCommit(BaseModel):
    """Request body for git commit."""

    message: str
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


class GitBranch(BaseModel):
    """Request body for git branch operations."""

    name: str
    start_point: Optional[str] = None


class GitPush(BaseModel):
    """Request body for git push."""

    remote: str = "origin"
    branch: Optional[str] = None
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


class GitPull(BaseModel):
    """Request body for git pull."""

    remote: str = "origin"
    branch: Optional[str] = None
    run_id: Optional[str] = None
    approval_id: Optional[str] = None


class LlmSelectionUpdate(BaseModel):
    """Request body for selecting active LLM company/model."""

    company_id: str
    model: str


class LlmAccessUpdate(BaseModel):
    """Request body for switching a provider on or off."""

    company_id: str
    enabled: bool


class CodexSettingsUpdate(BaseModel):
    """Request body for updating Codex-specific settings."""

    sandbox_mode: str


class LlmProviderInstallRequest(BaseModel):
    """Request body for installing a supported LLM CLI provider."""

    method: str = "brew"


class LlmProviderInstallJobStatus(BaseModel):
    """Status payload for an async LLM CLI provider install job."""

    job_id: str
    provider_id: str
    method: str
    status: str
    finished: bool
    returncode: Optional[int] = None
    installed: Optional[bool] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    output: str = ""


class IpLoginUpdate(BaseModel):
    """Request body for updating IP login setting."""

    allow_ip_login: bool


class PairVerifyRequest(BaseModel):
    """Request body for verifying a pairing token."""

    pair_token: str
    client_id: Optional[str] = None
    device_name: Optional[str] = None
    firebase_id_token: Optional[str] = None
    firebase_refresh_token: Optional[str] = None
    force_replace: bool = False  # If True, replace existing owner without confirmation


class PairCodeVerifyRequest(BaseModel):
    """Request body for verifying a 6-digit pairing code."""

    code: str


class SSOPairRequest(BaseModel):
    """Request body for SSO-based pairing (Firebase auth without QR).

    Used when app connects to a remote server via Firebase SSO.
    Server verifies the ID token and confirms ownership before issuing API key.
    """

    firebase_id_token: str
    firebase_refresh_token: Optional[str] = None
    client_id: Optional[str] = None
    device_name: Optional[str] = None
    force_replace: bool = False  # If True, replace existing owner without confirmation
