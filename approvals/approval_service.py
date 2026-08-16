"""Approval service that combines policy decisions, persistence, and audit."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from audit.audit_store import get_audit_store
from policy.policy_engine import default_policy_snapshot
from policy.policy_store import decide_policy_with_rules, get_policy_rule_store

from .approval_store import get_approval_store, is_request_expired

logger = logging.getLogger(__name__)

# How long a fresh approval stays answerable. A day is long enough that a
# prompt raised overnight is still there in the morning, and short enough that
# a run nobody ever answers stops occupying its task within a day instead of
# until the next server restart. Override with
# ``CODEBRIDGE_APPROVAL_EXPIRY_SECONDS``; ``0`` (or negative) means "never
# expires", which is an explicit opt-out and not the default because the
# default used to be, in effect, forever.
_DEFAULT_APPROVAL_EXPIRY_SECONDS = 24 * 60 * 60


def approval_expiry_seconds() -> int:
    raw = os.environ.get("CODEBRIDGE_APPROVAL_EXPIRY_SECONDS")
    if raw is None:
        return _DEFAULT_APPROVAL_EXPIRY_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_APPROVAL_EXPIRY_SECONDS
    return max(0, value)


def default_approval_expires_at(*, now: datetime | None = None) -> str | None:
    """When an approval created right now should stop being answerable.

    ``None`` when expiry is switched off, which callers pass straight through
    to ``expires_at=None`` — the pre-expiry behaviour.
    """
    seconds = approval_expiry_seconds()
    if seconds <= 0:
        return None
    moment = now or datetime.now(tz=timezone.utc)
    return (moment + timedelta(seconds=seconds)).isoformat()


def expire_approval(
    approval_id: str,
    *,
    trigger: str,
    attempted_decision: str | None = None,
    approver: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Flip a pending approval to ``expired`` and record that in the audit log.

    An expiry is a denial, so it has to leave the same kind of trail a denial
    does. ``trigger`` says who noticed — the periodic sweep, a resume that
    found a stale row, or an approver who pressed a button too late.
    """
    store = get_approval_store()
    request = store.get_request(approval_id)
    if not isinstance(request, dict):
        return None
    expired = store.mark_expired(approval_id) or request
    payload: dict[str, Any] = {
        "approval_id": approval_id,
        "expires_at": request.get("expires_at"),
        "trigger": trigger,
    }
    if attempted_decision:
        payload["attempted_decision"] = attempted_decision
    if approver is not None:
        payload["approver"] = approver
    get_audit_store().record_event(
        operation=request["operation"],
        run_id=request["run_id"],
        risk_level=request["risk_level"],
        decision="approval_expired",
        payload=payload,
    )
    return expired


def expire_stale_approvals() -> list[dict[str, Any]]:
    """Expire every pending approval whose deadline has passed.

    Returns the expired rows so a caller (the scheduler sweep) can settle the
    runs that were parked on them. Never raises: a broken row must not stop the
    rest of the sweep.
    """
    try:
        stale = get_approval_store().list_expired_pending()
    except Exception:
        logger.exception("could not list expired approvals")
        return []
    expired: list[dict[str, Any]] = []
    for request in stale:
        approval_id = request.get("id")
        if not isinstance(approval_id, str) or not approval_id:
            continue
        try:
            row = expire_approval(approval_id, trigger="sweep")
        except Exception:
            logger.exception("could not expire approval %s", approval_id)
            continue
        if isinstance(row, dict):
            expired.append(row)
    return expired


def request_approval_for_operation(
    *,
    operation: str,
    run_id: str | None = None,
    actor: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    risk_level: str | None = None,
    expires_at: str | None = None,
) -> dict[str, Any]:
    """Apply policy and create a pending approval when needed."""
    policy = decide_policy_with_rules(operation, run_id=run_id, details=details)
    audit = get_audit_store()
    if policy["forbidden"]:
        audit.record_event(
            operation=operation,
            run_id=run_id,
            risk_level=policy["risk_level"],
            decision="forbidden",
            payload={"actor": actor or {}, "details": details or {}, "policy": policy},
        )
        return {
            "approval_required": False,
            "allowed": False,
            "policy": policy,
            "error": policy["reason"],
        }

    if not policy["approval_required"]:
        audit.record_event(
            operation=operation,
            run_id=run_id,
            risk_level=policy["risk_level"],
            decision="allowed",
            payload={"actor": actor or {}, "details": details or {}, "policy": policy},
        )
        return {
            "approval_required": False,
            "allowed": True,
            "policy": policy,
        }

    request = get_approval_store().create_request(
        operation=operation,
        run_id=run_id,
        actor=actor or {},
        details=details or {},
        risk_level=risk_level or policy["risk_level"],
        expires_at=expires_at,
    )
    audit.record_event(
        operation=operation,
        run_id=run_id,
        risk_level=request["risk_level"],
        decision="approval_requested",
        payload={"actor": actor or {}, "details": details or {}, "policy": policy},
    )
    return {
        "approval_required": True,
        "allowed": False,
        "policy": policy,
        "approval": request,
    }


def standing_rule_scope_for_request(request: dict[str, Any]) -> str:
    """Narrowest policy scope a standing rule for ``request`` should use.

    Prefer the project the operation happened in, then the workspace, then the
    run. ``global`` is the last resort and is never chosen to *widen* an
    approval — a caller that wants it has to ask for it explicitly.
    """
    details = request.get("details") or {}
    if isinstance(details, dict):
        project_name = details.get("project_name")
        if isinstance(project_name, str) and project_name.strip() and project_name.strip() != "__global__":
            return f"project:{project_name.strip()}"
        workspace_id = details.get("workspace_id")
        if isinstance(workspace_id, str) and workspace_id.strip():
            return f"workspace:{workspace_id.strip()}"
    run_id = request.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return f"run:{run_id.strip()}"
    return "global"


def decide_approval(
    approval_id: str,
    *,
    decision: str,
    scope: str = "once",
    reason: str | None = None,
    constraints: dict[str, Any] | None = None,
    approver: dict[str, Any] | None = None,
    rule_scope: str | None = None,
    rule_expires_at: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a pending approval request and record an audit event.

    ``decision="approve_rule"`` additionally writes a standing ``allow`` policy
    rule, so the same operation stops prompting. That is what makes unattended
    (scheduled) runs possible: with no client connected there is nobody to
    answer a permission prompt, and the run would otherwise sit in
    ``waiting_for_user`` forever.
    """
    store = get_approval_store()
    request = store.get_request(approval_id)
    if request is None:
        return None

    # Reject decisions on requests whose expires_at has already passed.
    # We flip the row's status to ``expired`` so a stale request cannot
    # be approved later by the same approver clicking again, and emit a
    # dedicated audit event so reviewers can see expired-then-actioned
    # attempts in the timeline.
    if is_request_expired(request):
        expire_approval(
            approval_id,
            trigger="decision",
            attempted_decision=decision,
            approver=approver or {},
        )
        return {
            "error": "Approval request has expired",
            "approval": store.get_request(approval_id) or request,
        }

    if decision.startswith("approve") and request.get("desktop_only") and not _is_desktop_approver(approver):
        get_audit_store().record_event(
            operation=request["operation"],
            run_id=request["run_id"],
            risk_level=request["risk_level"],
            decision="approval_rejected",
            payload={
                "approval_id": approval_id,
                "reason": "Desktop-only approval requires the desktop app",
                "approver": approver or {},
            },
        )
        return {
            "error": "Desktop-only approval requires the desktop app",
            "approval": request,
        }

    result = store.create_decision(
        approval_id=approval_id,
        decision=decision,
        scope=scope,
        reason=reason,
        constraints=constraints or {},
        approver=approver or {},
    )
    if result is None:
        return None

    standing_rule: dict[str, Any] | None = None
    if decision == "approve_rule":
        standing_rule = get_policy_rule_store().create_rule(
            scope=rule_scope or standing_rule_scope_for_request(request),
            operation=request["operation"],
            effect="allow",
            constraints=constraints or {},
            created_by=_rule_author(approver),
            expires_at=rule_expires_at,
        )

    request = store.get_request(approval_id)
    get_audit_store().record_event(
        operation=request["operation"] if request else "approval.decision",
        run_id=request["run_id"] if request else None,
        risk_level=request["risk_level"] if request else None,
        decision=decision,
        payload={
            "approval_id": approval_id,
            "scope": scope,
            "reason": reason,
            "constraints": constraints or {},
            "approver": approver or {},
            # Recorded so an auditor can see the blast radius of the standing
            # rule this decision created, not just that it was approved once.
            "standing_rule": standing_rule,
        },
    )
    payload: dict[str, Any] = {"approval": request, "decision": result}
    if standing_rule is not None:
        payload["rule"] = standing_rule
    return payload


def _rule_author(approver: dict[str, Any] | None) -> str:
    if isinstance(approver, dict):
        for key in ("id", "name", "type"):
            value = approver.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "approval_decision"


def get_policy_snapshot() -> dict[str, Any]:
    """Return built-in policy information."""
    return {
        "default_policy": default_policy_snapshot(),
        "rules": get_policy_rule_store().list_rules(),
    }


def _is_desktop_approver(approver: dict[str, Any] | None) -> bool:
    if not approver:
        return False
    approver_type = str(approver.get("type") or "").strip().lower()
    return approver_type in {"desktop", "desktop_app", "local_desktop_app"}
