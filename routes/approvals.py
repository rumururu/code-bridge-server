"""Approval and policy APIs for Agent Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from approvals.approval_models import ApprovalDecisionCreate, ApprovalRequestCreate
from approvals.approval_service import (
    decide_approval,
    get_policy_snapshot,
    request_approval_for_operation,
)
from approvals.approval_store import get_approval_store

from .deps import verify_api_key

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


@router.post("/request", dependencies=[Depends(verify_api_key)], response_model=None)
async def request_approval(body: ApprovalRequestCreate) -> dict[str, Any] | JSONResponse:
    """Preflight an operation and create an approval request when required."""
    result = request_approval_for_operation(
        operation=body.operation,
        run_id=body.run_id,
        actor=body.actor,
        details=body.details,
        risk_level=body.risk_level,
        expires_at=body.expires_at,
    )
    if result.get("error"):
        return JSONResponse(status_code=403, content=result)
    return result


@router.get("/pending", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_pending_approvals(run_id: str | None = None) -> dict[str, Any]:
    """List pending approvals, optionally scoped to an agent run."""
    return {"approvals": get_approval_store().list_pending(run_id=run_id)}


@router.post("/{approval_id}/decision", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_approval_decision(
    approval_id: str,
    body: ApprovalDecisionCreate,
) -> dict[str, Any] | JSONResponse:
    """Approve or deny a pending approval request."""
    result = decide_approval(
        approval_id,
        decision=body.decision,
        scope=body.scope,
        reason=body.reason,
        constraints=body.constraints,
        approver=body.approver,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Approval '{approval_id}' not found")
    if result.get("error"):
        return JSONResponse(status_code=403, content=result)
    return result


@router.get("/policies", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_policies() -> dict[str, Any]:
    """Return built-in policy classes."""
    return get_policy_snapshot()
