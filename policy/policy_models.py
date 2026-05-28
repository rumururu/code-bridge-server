"""Pydantic models for persistent policy rules."""

from typing import Any

from pydantic import BaseModel, Field


class PolicyRuleCreate(BaseModel):
    """Request body for creating a persistent policy rule."""

    scope: str = "global"
    operation: str
    effect: str = Field(pattern="^(allow|confirm_once|confirm_each|desktop_only|forbidden)$")
    constraints: dict[str, Any] = Field(default_factory=dict)
    created_by: str | None = None
    expires_at: str | None = None
