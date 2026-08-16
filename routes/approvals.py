"""Approval and policy APIs for Agent Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from agent.approval_resume import maybe_resume_run_for_decision
from approvals.approval_models import ApprovalDecisionCreate, ApprovalRequestCreate
from approvals.approval_service import (
    decide_approval,
    get_policy_snapshot,
    request_approval_for_operation,
)
from approvals.approval_store import get_approval_store
from approvals.approver_channel import CHANNEL_REMOTE

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


async def apply_approval_decision(
    approval_id: str,
    body: ApprovalDecisionCreate,
    *,
    channel: str,
) -> dict[str, Any] | JSONResponse:
    """Resolve an approval and resume whatever run was parked on it.

    Both doors into this — the api-key route below and the dashboard mirror in
    :mod:`routes.dashboard_agents` — share this body so the resume hand-off and
    the error shapes cannot drift apart. ``channel`` is keyword-only and has
    **no default** on purpose: it is the one thing that decides whether a
    ``desktop_only`` approval may be granted, so a new caller has to state
    where its request came from rather than silently inheriting an answer.
    Pass one of the constants in :mod:`approvals.approver_channel`, chosen from
    the gate the route sits behind — never from the request body, which is
    exactly the mistake this replaces.
    """
    result = decide_approval(
        approval_id,
        decision=body.decision,
        scope=body.scope,
        reason=body.reason,
        constraints=body.constraints,
        approver=body.approver,
        channel=channel,
        rule_scope=body.rule_scope,
        rule_expires_at=body.rule_expires_at,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Approval '{approval_id}' not found")
    if result.get("error"):
        return JSONResponse(status_code=403, content=result)

    # Recording the decision is not the point of pressing approve — the run
    # carrying on is. If this approval is the one a run is parked on, hand the
    # decision to the orchestrator and let it continue in the background; the
    # response comes back immediately either way.
    resumed = await maybe_resume_run_for_decision(
        result.get("approval"), decision=body.decision
    )
    if resumed is not None:
        result = {**result, "resume": resumed}
    return result


@router.post("/{approval_id}/decision", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_approval_decision(
    approval_id: str,
    body: ApprovalDecisionCreate,
) -> dict[str, Any] | JSONResponse:
    """Approve or deny a pending approval request, from a paired client.

    This route is reachable with nothing but an API key — from the phone, and
    over the tunnel — so it is by definition not "someone at the desktop", no
    matter what ``body.approver`` says about itself. A ``desktop_only``
    approval arriving here is refused and audited as ``approval_rejected``;
    the dashboard mirror in :mod:`routes.dashboard_agents` is the door that
    can grant one.
    """
    return await apply_approval_decision(approval_id, body, channel=CHANNEL_REMOTE)


@router.get("/policies", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_policies() -> dict[str, Any]:
    """Return built-in policy classes."""
    return get_policy_snapshot()
