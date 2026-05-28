"""Pydantic models for approval APIs."""

from typing import Any

from pydantic import BaseModel, Field


class ApprovalRequestCreate(BaseModel):
    """Request body for creating or preflighting an approval request."""

    operation: str
    run_id: str | None = None
    actor: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    risk_level: str | None = None
    expires_at: str | None = None


class ApprovalDecisionCreate(BaseModel):
    """Request body for resolving an approval request."""

    decision: str = Field(pattern="^(approve_once|approve_for_session|approve_rule|deny|deny_with_instruction)$")
    scope: str = "once"
    reason: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    approver: dict[str, Any] = Field(default_factory=dict)

