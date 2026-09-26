"""SQLite-backed audit log for local Agent Cockpit operations."""

import hashlib
import json
import uuid
from typing import Any

from core.database import get_db_connection, init_db

from .audit_redactor import redact_payload


def _new_id() -> str:
    return f"aud_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _row_to_audit_event(row: Any) -> dict[str, Any]:
    # ``redacted_categories`` is a post-migration column; older rows / older
    # DBs may not expose it via sqlite3.Row, so guard the lookup.
    try:
        raw_categories = row["redacted_categories"]
    except (IndexError, KeyError):
        raw_categories = None
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "actor_type": row["actor_type"],
        "client_id": row["client_id"],
        "device_name": row["device_name"],
        "project_name": row["project_name"],
        "provider_id": row["provider_id"],
        "model": row["model"],
        "session_id": row["session_id"],
        "run_id": row["run_id"],
        "operation": row["operation"],
        "resource": row["resource"],
        "risk_level": row["risk_level"],
        "decision": row["decision"],
        "payload": _json_loads(row["payload_redacted_json"], {}),
        "affected_paths": _json_loads(row["affected_paths_json"], []),
        "redacted_categories": _json_loads(raw_categories, []),
        "hash_prev": row["hash_prev"],
        "hash_current": row["hash_current"],
    }


class AuditStore:
    """Local append-oriented audit storage."""

    def __init__(self) -> None:
        init_db()

    def record_event(
        self,
        *,
        operation: str,
        actor_type: str | None = None,
        client_id: str | None = None,
        device_name: str | None = None,
        project_name: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        resource: str | None = None,
        risk_level: str | None = None,
        decision: str | None = None,
        payload: dict[str, Any] | None = None,
        affected_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        event_id = _new_id()
        redacted_payload, redacted_categories = redact_payload(payload or {})
        affected_paths = affected_paths or []
        # Deterministic ordering so equality assertions in tests are stable
        # and identical category sets serialize to the same JSON string.
        categories_list = sorted(redacted_categories)
        categories_json = _json_dumps(categories_list) if categories_list else None
        with get_db_connection(use_row_factory=True) as conn:
            prev_hash_row = conn.execute(
                """
                SELECT hash_current FROM audit_events
                WHERE hash_current IS NOT NULL
                ORDER BY rowid DESC
                LIMIT 1
                """
            ).fetchone()
            prev_hash = prev_hash_row["hash_current"] if prev_hash_row else None
            hash_current = self._hash_event(
                event_id=event_id,
                operation=operation,
                payload=redacted_payload,
                affected_paths=affected_paths,
                prev_hash=prev_hash,
            )
            conn.execute(
                """
                INSERT INTO audit_events (
                    id, actor_type, client_id, device_name, project_name,
                    provider_id, model, session_id, run_id, operation, resource,
                    risk_level, decision, payload_redacted_json,
                    affected_paths_json, redacted_categories,
                    hash_prev, hash_current
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    actor_type,
                    client_id,
                    device_name,
                    project_name,
                    provider_id,
                    model,
                    session_id,
                    run_id,
                    operation,
                    resource,
                    risk_level,
                    decision,
                    _json_dumps(redacted_payload),
                    _json_dumps(affected_paths),
                    categories_json,
                    prev_hash,
                    hash_current,
                ),
            )
            conn.commit()
        return self.get_event(event_id) or {}

    def list_events(
        self,
        *,
        run_id: str | None = None,
        project_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            values.append(run_id)
        if project_name:
            clauses.append("project_name = ?")
            values.append(project_name)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 500)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM audit_events
                {where_sql}
                ORDER BY rowid DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_audit_event(row) for row in rows]

    def gated_operations_seen(self) -> set[str]:
        """Operation names the policy gate has actually been asked about.

        Only the decisions the gate itself records count — ``allowed``,
        ``approval_requested`` and ``forbidden`` (see
        ``approvals.approval_service.request_approval_for_operation``). A
        human's approve/deny is recorded under the same operation name, but
        that only proves a request row existed, not that it came through the
        gate: the feedback agent used to insert its requests straight into the
        table and the decisions on those were audited like any other.
        """
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT operation FROM audit_events
                WHERE decision IN ('allowed', 'approval_requested', 'forbidden')
                  AND operation IS NOT NULL AND operation != ''
                """
            ).fetchall()
        return {str(row["operation"]) for row in rows}

    def rule_match_stats(self) -> dict[str, dict[str, Any]]:
        """How often each standing policy rule has actually decided something.

        Keyed by rule id. Every gated operation records an audit event whose
        payload carries the policy decision, and when a persistent rule was
        the reason (``decide_policy_with_rules`` attaches it as
        ``policy.rule``), the rule's id is in ``payload.policy.rule.id``.
        Counting those rows is the only evidence a rule has ever been
        consulted — the rules table itself says nothing about use, and a rule
        for an operation no code path gates looks identical to one that fires
        every night. The dashboard shows this next to each rule for exactly
        that reason.
        """
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT
                    json_extract(payload_redacted_json, '$.policy.rule.id') AS rule_id,
                    COUNT(*) AS matched_count,
                    MAX(timestamp) AS last_matched_at
                FROM audit_events
                WHERE json_extract(payload_redacted_json, '$.policy.rule.id') IS NOT NULL
                GROUP BY rule_id
                """
            ).fetchall()
        return {
            str(row["rule_id"]): {
                "matched_count": int(row["matched_count"] or 0),
                "last_matched_at": row["last_matched_at"],
            }
            for row in rows
            if row["rule_id"]
        }

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM audit_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return _row_to_audit_event(row) if row else None

    def _hash_event(
        self,
        *,
        event_id: str,
        operation: str,
        payload: dict[str, Any],
        affected_paths: list[str],
        prev_hash: str | None,
    ) -> str:
        digest_input = _json_dumps(
            {
                "event_id": event_id,
                "operation": operation,
                "payload": payload,
                "affected_paths": affected_paths,
                "prev_hash": prev_hash,
            }
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


_audit_store: AuditStore | None = None


def get_audit_store() -> AuditStore:
    """Return the process-global audit store."""
    global _audit_store
    if _audit_store is None:
        _audit_store = AuditStore()
    return _audit_store
