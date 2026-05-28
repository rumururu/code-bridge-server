"""Optional policy/approval gate for direct API tool actions."""

from typing import Any

from approvals.approval_service import request_approval_for_operation
from approvals.approval_store import get_approval_store


def evaluate_direct_action_gate(
    *,
    operation: str,
    project_name: str | None = None,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
    require_approval: bool = False,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Return whether a direct action may execute.

    Existing app routes keep working with ``require_approval=False``. Cockpit
    callers can set ``require_approval=True`` to force the approval lifecycle
    before running a mutating or executing action.
    """
    details = details or {}
    if project_name:
        details = {"project_name": project_name, **details}

    if approval_id:
        request = get_approval_store().get_request(approval_id)
        if request is None:
            return {
                "allowed": False,
                "status_code": 404,
                "payload": {"error": f"Approval '{approval_id}' not found"},
            }
        if request["operation"] != operation:
            return {
                "allowed": False,
                "status_code": 409,
                "payload": {
                    "error": "Approval operation does not match requested action",
                    "approval": request,
                },
            }
        if run_id and request.get("run_id") and request["run_id"] != run_id:
            return {
                "allowed": False,
                "status_code": 409,
                "payload": {
                    "error": "Approval run does not match requested action",
                    "approval": request,
                },
            }
        if request["status"] != "approved":
            return {
                "allowed": False,
                "status_code": 409 if request["status"] == "pending" else 403,
                "payload": {
                    "error": f"Approval is {request['status']}",
                    "approval": request,
                },
            }
        return {"allowed": True, "approval": request}

    if not require_approval:
        return {"allowed": True}

    result = request_approval_for_operation(
        operation=operation,
        run_id=run_id,
        actor={"type": "api_client"},
        details=details,
    )
    if result.get("error"):
        return {"allowed": False, "status_code": 403, "payload": result}
    if result.get("approval_required"):
        return {"allowed": False, "status_code": 409, "payload": result}
    return {"allowed": True, "policy": result.get("policy")}
