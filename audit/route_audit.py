"""Helpers for auditing direct API actions."""

from typing import Any

from policy.policy_store import decide_policy_with_rules

from .audit_store import get_audit_store


def record_api_action(
    *,
    operation: str,
    project_name: str | None = None,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
    success: bool,
    status_code: int,
) -> None:
    """Best-effort audit for user/API initiated actions."""
    details = details or {}
    policy = decide_policy_with_rules(operation, run_id=run_id, details=details)
    decision = "executed" if success else "failed"
    try:
        get_audit_store().record_event(
            operation=operation,
            actor_type="api_client",
            project_name=project_name,
            run_id=run_id,
            risk_level=policy["risk_level"],
            decision=decision,
            payload={
                "details": details,
                "status_code": status_code,
                "policy": policy,
            },
        )
    except Exception:
        # Route audits must never change the result of the user action.
        return
