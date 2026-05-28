"""SQLite persistence for approval requests and decisions."""

import json
import uuid
from typing import Any

from core.database import get_db_connection, init_db
from policy.policy_store import decide_policy_with_rules


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _row_to_request(row: Any) -> dict[str, Any]:
    details = _json_loads(row["details_json"], {})
    policy = decide_policy_with_rules(
        row["operation"],
        run_id=row["run_id"],
        details=details,
    )
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "actor": _json_loads(row["actor_json"], {}),
        "operation": row["operation"],
        "risk_level": row["risk_level"],
        "details": details,
        "policy": policy,
        "desktop_only": bool(policy.get("desktop_only")),
        "status": row["status"],
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
        "resolved_at": row["resolved_at"],
    }


def _row_to_decision(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "approval_id": row["approval_id"],
        "decision": row["decision"],
        "scope": row["scope"],
        "reason": row["reason"],
        "constraints": _json_loads(row["constraints_json"], {}),
        "approver": _json_loads(row["approver_json"], {}),
        "created_at": row["created_at"],
    }


class ApprovalStore:
    """Persistence helper for approval lifecycle records."""

    def __init__(self) -> None:
        init_db()

    def create_request(
        self,
        *,
        operation: str,
        run_id: str | None = None,
        actor: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        risk_level: str = "medium",
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        approval_id = _new_id("apr")
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO approval_requests (
                    id, run_id, actor_json, operation, risk_level,
                    details_json, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    run_id,
                    _json_dumps(actor or {}),
                    operation,
                    risk_level,
                    _json_dumps(details or {}),
                    expires_at,
                ),
            )
            conn.commit()
        return self.get_request(approval_id) or {}

    def get_request(self, approval_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM approval_requests WHERE id = ?",
                (approval_id,),
            ).fetchone()
        return _row_to_request(row) if row else None

    def list_pending(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        clauses = ["status = 'pending'"]
        values: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            values.append(run_id)
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM approval_requests
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at ASC
                """,
                values,
            ).fetchall()
        return [_row_to_request(row) for row in rows]

    def create_decision(
        self,
        *,
        approval_id: str,
        decision: str,
        scope: str = "once",
        reason: str | None = None,
        constraints: dict[str, Any] | None = None,
        approver: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        request = self.get_request(approval_id)
        if not request:
            return None
        decision_id = _new_id("dec")
        status = "approved" if decision.startswith("approve") else "denied"
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO approval_decisions (
                    id, approval_id, decision, scope, reason,
                    constraints_json, approver_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    approval_id,
                    decision,
                    scope,
                    reason,
                    _json_dumps(constraints or {}),
                    _json_dumps(approver or {}),
                ),
            )
            conn.execute(
                """
                UPDATE approval_requests
                SET status = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, approval_id),
            )
            conn.commit()
        return self.get_decision(decision_id)

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM approval_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
        return _row_to_decision(row) if row else None


_approval_store: ApprovalStore | None = None


def get_approval_store() -> ApprovalStore:
    """Return the process-global approval store."""
    global _approval_store
    if _approval_store is None:
        _approval_store = ApprovalStore()
    return _approval_store
