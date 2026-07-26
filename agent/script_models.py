"""Pydantic models for the registered-script APIs."""

from typing import Any

from pydantic import BaseModel, Field


class ScriptRegister(BaseModel):
    """Request body for registering a script a shell step may run."""

    name: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1)
    description: str | None = None
    interpreter: str = "bash"
    default_args: list[str] = Field(default_factory=list)
    timeout_seconds: int | None = None
    created_by: str | None = None


class ScriptUpdate(BaseModel):
    """Partial update. Only the fields present are changed."""

    name: str | None = Field(default=None, min_length=1, max_length=80)
    path: str | None = None
    description: str | None = None
    interpreter: str | None = None
    default_args: list[str] | None = None
    timeout_seconds: int | None = None
