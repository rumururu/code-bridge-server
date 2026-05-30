"""SQLite persistence for Agent Cockpit runs, tasks, events, and artifacts.

Row -> dict converters live in :mod:`agent._row_converters` so this module
can focus on query/mutation logic.
"""

import json
import uuid
from typing import Any

from core.database import get_db_connection, init_db

from ._capability_store import CapabilityStoreMixin
from ._connector_request_store import ConnectorRequestStoreMixin
from ._row_converters import (
    _json_loads,
    _row_to_artifact,
    _row_to_capability,
    _row_to_connector_request,
    _row_to_event,
    _row_to_message,
    _row_to_run,
    _row_to_task,
    _row_to_task_capability,
    _row_to_task_run,
    _row_to_task_step,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class AgentStore(CapabilityStoreMixin, ConnectorRequestStoreMixin):
    """Persistence helper for durable agent platform records.

    Domain-specific groups of methods are split into mixins under
    ``server/agent/_*_store.py`` and composed here. ``AgentStore`` itself
    keeps the runs / messages / events / artifacts / tasks / task-steps /
    task-capabilities / capability-registry methods that share a lot of
    cross-table state and are too tightly coupled to extract safely yet.
    """

    def __init__(self) -> None:
        init_db()

    def create_run(
        self,
        *,
        project_name: str | None = None,
        workspace_id: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
        title: str | None = None,
        goal: str | None = None,
        cwd: str | None = None,
        parent_run_id: str | None = None,
        native_session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = _new_id("run")
        status = "queued"
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    id, workspace_id, project_name, task_id, provider_id, model, status,
                    title, goal, cwd, native_session_id, parent_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    workspace_id,
                    project_name,
                    task_id,
                    provider_id,
                    model,
                    status,
                    title,
                    goal,
                    cwd,
                    native_session_id,
                    parent_run_id,
                ),
            )
            conn.commit()
        if task_id:
            self.link_task_run(task_id=task_id, run_id=run_id, role="primary")
        return self.get_run(run_id) or {}

    def list_runs(
        self,
        *,
        project_name: str | None = None,
        workspace_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if workspace_id:
            clauses.append("workspace_id = ?")
            values.append(workspace_id)
        if project_name:
            clauses.append("project_name = ?")
            values.append(project_name)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 200)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM agent_runs
                {where_sql}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_run(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return _row_to_run(row) if row else None

    def update_run_status(self, run_id: str, status: str) -> dict[str, Any] | None:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE agent_runs
                SET status = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    ended_at = CASE
                        WHEN ? IN ('completed', 'failed', 'cancelled') THEN CURRENT_TIMESTAMP
                        ELSE ended_at
                    END
                WHERE id = ?
                """,
                (status, status, run_id),
            )
            conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_run(run_id)

    def cancel_active_runs(
        self,
        *,
        statuses: tuple[str, ...] = ("queued", "running"),
    ) -> list[dict[str, Any]]:
        """Mark queued/running runs as cancelled and return updated rows."""
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT id FROM agent_runs
                WHERE status IN ({placeholders})
                """,
                list(statuses),
            ).fetchall()
            run_ids = [row["id"] for row in rows]
            if run_ids:
                id_placeholders = ",".join("?" for _ in run_ids)
                conn.execute(
                    f"""
                    UPDATE agent_runs
                    SET status = 'cancelled',
                        updated_at = CURRENT_TIMESTAMP,
                        ended_at = CURRENT_TIMESTAMP
                    WHERE id IN ({id_placeholders})
                    """,
                    run_ids,
                )
            conn.commit()
        return [run for run_id in run_ids if (run := self.get_run(run_id)) is not None]

    def add_message(
        self,
        *,
        run_id: str,
        role: str,
        content: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if self.get_run(run_id) is None:
            return None
        message_id = _new_id("msg")
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_messages (id, run_id, role, content, attachments_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    run_id,
                    role,
                    content,
                    _json_dumps(attachments or []),
                ),
            )
            conn.execute(
                "UPDATE agent_runs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,),
            )
            conn.commit()
        return self.get_message(message_id)

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return _row_to_message(row) if row else None

    def list_messages(self, run_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_messages
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [_row_to_message(row) for row in rows]

    def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        provider_id: str | None = None,
        provider_event: dict[str, Any] | None = None,
        app_event: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.get_run(run_id) is None:
            return None
        event_id = _new_id("evt")
        with get_db_connection() as conn:
            next_sequence = int(
                conn.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM agent_events
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """
                INSERT INTO agent_events (
                    id, run_id, sequence, event_type, provider_id,
                    provider_event_json, app_event_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    run_id,
                    next_sequence,
                    event_type,
                    provider_id,
                    _json_dumps(provider_event) if provider_event is not None else None,
                    _json_dumps(app_event) if app_event is not None else None,
                ),
            )
            conn.execute(
                "UPDATE agent_runs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,),
            )
            conn.commit()
        return self.get_event(event_id)

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return _row_to_event(row) if row else None

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if after_sequence is not None:
            clauses.append("sequence > ?")
            values.append(after_sequence)
        values.append(max(1, min(int(limit), 500)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM agent_events
                WHERE {' AND '.join(clauses)}
                ORDER BY sequence ASC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_event(row) for row in rows]

    def add_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        path: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.get_run(run_id) is None:
            return None
        artifact_id = _new_id("art")
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_artifacts (id, run_id, kind, path, mime_type, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    run_id,
                    kind,
                    path,
                    mime_type,
                    _json_dumps(metadata or {}),
                ),
            )
            conn.execute(
                "UPDATE agent_runs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,),
            )
            conn.commit()
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()
        return _row_to_artifact(row) if row else None

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_artifacts
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [_row_to_artifact(row) for row in rows]

    def create_task(
        self,
        *,
        title: str,
        description: str | None = None,
        project_name: str | None = None,
        workspace_id: str | None = None,
        run_id: str | None = None,
        kind: str = "general",
        source: str = "manual",
        goal: str | None = None,
        priority: int = 0,
        due_at: str | None = None,
        labels: list[str] | None = None,
        assignee: str | None = None,
        requester: str | None = None,
        acceptance: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_id = _new_id("task")
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_tasks (
                    id, workspace_id, run_id, project_name, kind, source,
                    title, description, goal, priority, due_at, labels_json,
                    assignee, requester, acceptance_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    workspace_id,
                    run_id,
                    project_name,
                    kind or "general",
                    source or "manual",
                    title,
                    description,
                    goal,
                    int(priority),
                    due_at,
                    _json_dumps(labels or []),
                    assignee,
                    requester,
                    _json_dumps(acceptance or []),
                    _json_dumps(metadata or {}),
                ),
            )
            conn.commit()
        if run_id:
            self.link_task_run(task_id=task_id, run_id=run_id, role="primary")
        return self.get_task(task_id) or {}

    def list_tasks(
        self,
        *,
        workspace_id: str | None = None,
        project_name: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if workspace_id:
            clauses.append("workspace_id = ?")
            values.append(workspace_id)
        if project_name:
            clauses.append("project_name = ?")
            values.append(project_name)
        if kind:
            clauses.append("kind = ?")
            values.append(kind)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 200)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM agent_tasks
                {where_sql}
                ORDER BY priority DESC, updated_at DESC, created_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_task(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        return _row_to_task(row) if row else None

    def update_task(
        self,
        task_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        scalar_columns = {
            "title",
            "description",
            "project_name",
            "workspace_id",
            "run_id",
            "kind",
            "source",
            "goal",
            "status",
            "priority",
            "due_at",
            "assignee",
            "requester",
        }
        json_columns = {
            "labels": "labels_json",
            "acceptance": "acceptance_json",
            "metadata": "metadata_json",
            "result": "result_json",
            "error": "error_json",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in updates.items():
            if key in scalar_columns:
                assignments.append(f"{key} = ?")
                values.append(int(value) if key == "priority" and value is not None else value)
            elif key in json_columns:
                assignments.append(f"{json_columns[key]} = ?")
                values.append(_json_dumps(value or ([] if key in {"labels", "acceptance"} else {})))
        status = updates.get("status")
        if status in {"queued", "running", "in_progress"}:
            assignments.append("started_at = COALESCE(started_at, CURRENT_TIMESTAMP)")
        if status in {"completed", "done", "failed", "cancelled"}:
            assignments.append("ended_at = COALESCE(ended_at, CURRENT_TIMESTAMP)")
        if not assignments:
            return self.get_task(task_id)
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(task_id)
        with get_db_connection() as conn:
            cursor = conn.execute(
                f"""
                UPDATE agent_tasks
                SET {', '.join(assignments)}
                WHERE id = ?
                """,
                values,
            )
            conn.commit()
        if cursor.rowcount == 0:
            return None
        updated = self.get_task(task_id)
        if updated and updates.get("run_id"):
            self.link_task_run(task_id=task_id, run_id=str(updates["run_id"]), role="primary")
        return updated

    def update_task_status(self, task_id: str, status: str) -> dict[str, Any] | None:
        return self.update_task(task_id, {"status": status})

    def link_task_run(
        self,
        *,
        task_id: str,
        run_id: str,
        role: str = "primary",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.get_task(task_id) is None or self.get_run(run_id) is None:
            return None
        link_id = _new_id("trun")
        with get_db_connection(use_row_factory=True) as conn:
            next_sequence = int(
                conn.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM agent_task_runs
                    WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_task_runs (
                    id, task_id, run_id, role, sequence, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    link_id,
                    task_id,
                    run_id,
                    role or "primary",
                    next_sequence,
                    _json_dumps(metadata or {}),
                ),
            )
            conn.execute(
                """
                UPDATE agent_runs
                SET task_id = COALESCE(task_id, ?), updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (task_id, run_id),
            )
            conn.execute(
                """
                UPDATE agent_tasks
                SET run_id = COALESCE(run_id, ?), updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (run_id, task_id),
            )
            conn.commit()
            row = conn.execute(
                """
                SELECT * FROM agent_task_runs
                WHERE task_id = ? AND run_id = ?
                """,
                (task_id, run_id),
            ).fetchone()
        return _row_to_task_run(row) if row else None

    def list_task_runs(self, task_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_task_runs
                WHERE task_id = ?
                ORDER BY sequence ASC, created_at ASC
                """,
                (task_id,),
            ).fetchall()
        return [_row_to_task_run(row) for row in rows]

    def create_task_step(
        self,
        *,
        task_id: str,
        title: str | None = None,
        run_id: str | None = None,
        capability_id: str | None = None,
        status: str = "queued",
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        approval_id: str | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self.get_task(task_id) is None:
            return None
        step_id = _new_id("step")
        with get_db_connection(use_row_factory=True) as conn:
            next_sequence = int(
                conn.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM agent_task_steps
                    WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """
                INSERT INTO agent_task_steps (
                    id, task_id, run_id, capability_id, sequence, title, status,
                    input_json, output_json, approval_id, artifact_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    task_id,
                    run_id,
                    capability_id,
                    next_sequence,
                    title,
                    status,
                    _json_dumps(input or {}),
                    _json_dumps(output or {}),
                    approval_id,
                    artifact_id,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_task_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
        return _row_to_task_step(row) if row else None

    def list_task_steps(self, task_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_task_steps
                WHERE task_id = ?
                ORDER BY sequence ASC
                """,
                (task_id,),
            ).fetchall()
        return [_row_to_task_step(row) for row in rows]

    def get_task_step(self, step_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_task_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
        return _row_to_task_step(row) if row else None

    def update_task_step(
        self,
        step_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        scalar_columns = {"run_id", "capability_id", "title", "status", "approval_id", "artifact_id"}
        json_columns = {"input": "input_json", "output": "output_json"}
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
        if status in {"running", "in_progress"}:
            assignments.append("started_at = COALESCE(started_at, CURRENT_TIMESTAMP)")
        if status in {"completed", "done", "failed", "blocked", "cancelled"}:
            assignments.append("ended_at = COALESCE(ended_at, CURRENT_TIMESTAMP)")
        if not assignments:
            with get_db_connection(use_row_factory=True) as conn:
                row = conn.execute(
                    "SELECT * FROM agent_task_steps WHERE id = ?",
                    (step_id,),
                ).fetchone()
            return _row_to_task_step(row) if row else None
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(step_id)
        with get_db_connection(use_row_factory=True) as conn:
            cursor = conn.execute(
                f"""
                UPDATE agent_task_steps
                SET {', '.join(assignments)}
                WHERE id = ?
                """,
                values,
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                "SELECT * FROM agent_task_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
        return _row_to_task_step(row) if row else None

    def link_task_capability(
        self,
        *,
        task_id: str,
        capability_id: str,
        mode: str = "read",
    ) -> dict[str, Any] | None:
        if self.get_task(task_id) is None or self.get_capability(capability_id) is None:
            return None
        link_id = _new_id("tcap")
        resolved_mode = mode or "read"
        with get_db_connection(use_row_factory=True) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_task_capabilities (
                    id, task_id, capability_id, mode
                )
                VALUES (?, ?, ?, ?)
                """,
                (link_id, task_id, capability_id, resolved_mode),
            )
            conn.commit()
            row = conn.execute(
                """
                SELECT * FROM agent_task_capabilities
                WHERE task_id = ? AND capability_id = ? AND mode = ?
                """,
                (task_id, capability_id, resolved_mode),
            ).fetchone()
        return _row_to_task_capability(row) if row else None

    def list_task_capabilities(self, task_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_task_capabilities
                WHERE task_id = ?
                ORDER BY created_at ASC
                """,
                (task_id,),
            ).fetchall()
        return [_row_to_task_capability(row) for row in rows]

    def list_task_timeline(self, task_id: str) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if task is None:
            return None
        run_links = self.list_task_runs(task_id)
        run_ids = [link["run_id"] for link in run_links]
        runs = [run for run_id in run_ids if (run := self.get_run(run_id)) is not None]
        events: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        for run_id in run_ids:
            events.extend(self.list_events(run_id))
            artifacts.extend(self.list_artifacts(run_id))
        return {
            "task": task,
            "runs": runs,
            "run_links": run_links,
            "steps": self.list_task_steps(task_id),
            "capability_links": self.list_task_capabilities(task_id),
            "connector_requests": self.list_connector_requests(task_id=task_id, limit=200),
            "events": events,
            "artifacts": artifacts,
        }


_agent_store: AgentStore | None = None


def get_agent_store() -> AgentStore:
    """Return the process-global agent store."""
    global _agent_store
    if _agent_store is None:
        _agent_store = AgentStore()
    return _agent_store
