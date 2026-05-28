"""Persistent policy rule APIs for Agent Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from policy.policy_models import PolicyRuleCreate
from policy.policy_store import get_policy_rule_store

from .deps import verify_api_key

router = APIRouter(prefix="/api/policies", tags=["policies"])


@router.get("/rules", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_policy_rules(
    scope: str | None = None,
    operation: str | None = None,
    include_expired: bool = False,
) -> dict[str, Any]:
    """List persistent policy rules."""
    return {
        "rules": get_policy_rule_store().list_rules(
            scope=scope,
            operation=operation,
            include_expired=include_expired,
        )
    }


@router.post("/rules", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_policy_rule(body: PolicyRuleCreate) -> dict[str, Any]:
    """Create a persistent policy rule."""
    return {
        "rule": get_policy_rule_store().create_rule(
            scope=body.scope,
            operation=body.operation,
            effect=body.effect,
            constraints=body.constraints,
            created_by=body.created_by,
            expires_at=body.expires_at,
        )
    }


@router.delete("/rules/{rule_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def delete_policy_rule(rule_id: str) -> dict[str, Any]:
    """Delete a persistent policy rule."""
    deleted = get_policy_rule_store().delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Policy rule '{rule_id}' not found")
    return {"deleted": True}
