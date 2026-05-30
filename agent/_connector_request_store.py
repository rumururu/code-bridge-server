"""Connector-request persistence mixin for AgentStore.

Connector requests are durable handles for deferred skill / MCP / native-
provider tool invocations created by the task orchestrator when a step's
adapter requires explicit human review before execution. They live in
``agent_connector_requests`` and have their own create / read / update
verbs that the orchestrator and cockpit drive separately from the main
run/event flow.

This module owns those four methods. ``AgentStore`` mixes the class in
so callers continue to go through ``get_agent_store().create_connector_request(...)``
unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.database import get_db_connection

from ._row_converters import _row_to_connector_request


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class ConnectorRequestStoreMixin:
    """Mixin contributing connector-request methods to AgentStore.

    Depends on ``self.get_task`` and ``self.get_connector_request`` being
    provided by the surrounding store class.
    """

    def create_connector_request(
        self,
        *,
        task_id: str,
        connector_type: str,
        name: str,
        step_id: str | None = None,
        run_id: str | None = None,
        status: str = "pending_review",
        adapter: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.get_task(task_id) is None:  # type: ignore[attr-defined]
            return None
        request_id = _new_id("creq")
        with get_db_connection(use_row_factory=True) as conn:
            conn.execute(
                """
                INSERT INTO agent_connector_requests (
                    id, task_id, step_id, run_id, connector_type, name, status,
                    adapter_json, parameters_json, result_json, error_json,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CASE
                    WHEN ? IN ('completed', 'failed', 'cancelled') THEN CURRENT_TIMESTAMP
                    ELSE NULL
                END)
                """,
                (
                    request_id,
                    task_id,
                    step_id,
                    run_id,
                    connector_type,
                    name,
                    status or "pending_review",
                    _json_dumps(adapter or {}),
                    _json_dumps(parameters or {}),
                    _json_dumps(result or {}),
                    _json_dumps(error or {}),
                    status or "pending_review",
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_connector_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
        return _row_to_connector_request(row) if row else None

    def get_connector_request(self, request_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_connector_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
        return _row_to_connector_request(row) if row else None

    def list_connector_requests(
        self,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if task_id:
            clauses.append("task_id = ?")
            values.append(task_id)
        if step_id:
            clauses.append("step_id = ?")
            values.append(step_id)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 500)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM agent_connector_requests
                {where_sql}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_connector_request(row) for row in rows]

    def update_connector_request(
        self,
        request_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        scalar_columns = {"status", "connector_type", "name", "step_id", "run_id"}
        json_columns = {
            "adapter": "adapter_json",
            "parameters": "parameters_json",
            "result": "result_json",
            "error": "error_json",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in updates.items():
            if key in scalar_columns:
                assignments.append(f"{key} = ?")
                values.append(value)
            elif key in json_columns:
                assignments.append(f"{json_columns[key]} = ?")
                values.append(_json_dumps(value or {}))
        status = updates.get("status")
        if status in {"completed", "failed", "cancelled"}:
            assignments.append(
                "completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)"
            )
        if not assignments:
            return self.get_connector_request(request_id)
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(request_id)
        with get_db_connection(use_row_factory=True) as conn:
            cursor = conn.execute(
                f"""
                UPDATE agent_connector_requests
                SET {', '.join(assignments)}
                WHERE id = ?
                """,
                values,
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                "SELECT * FROM agent_connector_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
        return _row_to_connector_request(row) if row else None
