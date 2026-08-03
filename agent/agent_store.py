"""SQLite persistence for Agent Cockpit runs, tasks, events, and artifacts.

Row -> dict converters live in :mod:`agent._row_converters` so this module
can focus on query/mutation logic.
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from core.database import get_db_connection, init_db

from ._artifact_store import ArtifactStoreMixin
from ._capability_store import CapabilityStoreMixin
from ._connector_request_store import ConnectorRequestStoreMixin
from ._event_store import EventStoreMixin
from ._row_converters import (
    _row_to_agent,
    _row_to_memory,
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


def _json_field(value: Any, default: Any) -> str:
    return _json_dumps(default if value is None else value)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _row_to_run_record(row: Any) -> dict[str, Any]:
    run = _row_to_run(row)
    keys = row.keys()
    run["dry_run"] = bool(row["dry_run"]) if "dry_run" in keys else False
    run["simulated_summary"] = (
        _json_loads(row["simulated_summary_json"], None)
        if "simulated_summary_json" in keys
        else None
    )
    return run


logger = logging.getLogger(__name__)

_CHECKPOINT_STEP_STATUSES = {"waiting_for_user", "blocked"}


def _checkpoint_from_step(step: dict[str, Any]) -> dict[str, Any] | None:
    output = step.get("output")
    if not isinstance(output, dict):
        output = {}
    checkpoint = output.get("checkpoint")
    if isinstance(checkpoint, dict):
        return checkpoint

    step_input = step.get("input")
    if not isinstance(step_input, dict):
        step_input = {}
    workflow_step_id = step_input.get("workflow_step_id")
    return {
        "status": step.get("status"),
        "reason": output.get("reason") or step.get("status"),
        "prompt": output.get("prompt") or output.get("message") or "",
        "workflow_step_id": workflow_step_id if isinstance(workflow_step_id, str) else None,
        "resume": "same_step",
        "resume_step_id": None,
        "required_user_action": output.get("required_user_action") or "",
        "created_at": step.get("updated_at") or step.get("created_at"),
    }


class AgentStoreConflictError(Exception):
    """Raised when a requested store mutation conflicts with active state."""


class PseudoAgentProtectedError(Exception):
    """Raised when a mutation targets a protected pseudo-agent."""


class AgentStore(
    ArtifactStoreMixin,
    CapabilityStoreMixin,
    ConnectorRequestStoreMixin,
    EventStoreMixin,
):
    """Persistence helper for durable agent platform records.

    Domain-specific groups of methods are split into mixins under
    ``server/agent/_*_store.py`` and composed here. ``AgentStore`` itself
    keeps the runs / messages / events / artifacts / tasks / task-steps /
    task-capabilities / capability-registry methods that share a lot of
    cross-table state and are too tightly coupled to extract safely yet.
    """

    def __init__(self) -> None:
        init_db()

    def create_agent(
        self,
        *,
        name: str,
        description: str | None = None,
        system_prompt: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
        tools_json: Any = None,
        flow_json: Any = None,
        policy_overrides_json: Any = None,
    ) -> dict[str, Any]:
        agent_id = _new_id("agent")
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agents (
                    id, name, description, system_prompt, provider_id, model,
                    tools_json, flow_json, policy_overrides_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    name,
                    description,
                    system_prompt or "",
                    provider_id,
                    model,
                    _json_field(tools_json, []),
                    _json_field(flow_json, []),
                    _json_field(policy_overrides_json, {}),
                ),
            )
            conn.commit()
        return self.get_agent(agent_id) or {}

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
        return _row_to_agent(row) if row else None

    def list_agents(
        self,
        *,
        include_archived: bool = False,
        include_pseudo: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        if not include_archived:
            clauses.append("archived_at IS NULL")
        if not include_pseudo:
            clauses.append("is_pseudo = 0")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM agents
                {where_sql}
                ORDER BY is_pseudo ASC, updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [_row_to_agent(row) for row in rows]

    def count_agents(
        self,
        *,
        include_archived: bool = False,
        include_pseudo: bool = False,
    ) -> int:
        clauses: list[str] = []
        if not include_archived:
            clauses.append("archived_at IS NULL")
        if not include_pseudo:
            clauses.append("is_pseudo = 0")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with get_db_connection() as conn:
            return int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM agents
                    {where_sql}
                    """
                ).fetchone()[0]
            )

    def get_agent_activation_summary(self, agent_id: str) -> dict[str, Any]:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT t.id) AS assigned_task_count,
                    COUNT(DISTINCT CASE WHEN s.id IS NOT NULL THEN t.id END)
                        AS scheduled_task_count,
                    COUNT(s.id) AS schedule_count,
                    COALESCE(SUM(CASE WHEN s.enabled = 1 THEN 1 ELSE 0 END), 0)
                        AS enabled_schedule_count
                FROM agent_tasks t
                LEFT JOIN task_schedules s ON s.task_id = t.id
                WHERE t.assigned_agent_id = ?
                """,
                (agent_id,),
            ).fetchone()

        assigned_task_count = int(row["assigned_task_count"] or 0) if row else 0
        scheduled_task_count = int(row["scheduled_task_count"] or 0) if row else 0
        schedule_count = int(row["schedule_count"] or 0) if row else 0
        enabled_schedule_count = int(row["enabled_schedule_count"] or 0) if row else 0
        return {
            "has_assigned_tasks": assigned_task_count > 0,
            "has_schedules": schedule_count > 0,
            "has_enabled_schedules": enabled_schedule_count > 0,
            "assigned_task_count": assigned_task_count,
            "scheduled_task_count": scheduled_task_count,
            "schedule_count": schedule_count,
            "enabled_schedule_count": enabled_schedule_count,
        }

    def update_agent(
        self,
        agent_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        current = self.get_agent(agent_id)
        if current is None:
            return None
        if current.get("is_pseudo"):
            raise PseudoAgentProtectedError("pseudo_agent_protected")

        scalar_columns = {
            "name",
            "description",
            "system_prompt",
            "provider_id",
            "model",
        }
        json_columns = {
            "tools_json": [],
            "flow_json": [],
            "policy_overrides_json": {},
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in updates.items():
            if key in scalar_columns:
                assignments.append(f"{key} = ?")
                values.append(value)
            elif key in json_columns:
                assignments.append(f"{key} = ?")
                values.append(_json_field(value, json_columns[key]))
        if not assignments:
            return current

        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(agent_id)
        with get_db_connection() as conn:
            cursor = conn.execute(
                f"""
                UPDATE agents
                SET {', '.join(assignments)}
                WHERE id = ?
                """,
                values,
            )
            conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_agent(agent_id)

    def archive_agent(self, agent_id: str, *, archive: bool = True) -> dict[str, Any] | None:
        current = self.get_agent(agent_id)
        if current is None:
            return None
        if current.get("is_pseudo"):
            raise PseudoAgentProtectedError("pseudo_agent_protected")

        with get_db_connection() as conn:
            if archive:
                conn.execute(
                    """
                    UPDATE task_schedules
                    SET enabled = 0,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE task_id IN (
                        SELECT id FROM agent_tasks
                        WHERE assigned_agent_id = ?
                    )
                    """,
                    (agent_id,),
                )
                conn.execute(
                    """
                    UPDATE agents
                    SET archived_at = COALESCE(archived_at, CURRENT_TIMESTAMP),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (agent_id,),
                )
                conn.commit()
                return {"id": agent_id, "archived": True}

            active_run_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM agent_runs
                    WHERE agent_id = ? AND status IN ('queued', 'running')
                    """,
                    (agent_id,),
                ).fetchone()[0]
            )
            if active_run_count > 0:
                raise AgentStoreConflictError("active_agent_runs")
            conn.execute(
                """
                UPDATE task_schedules
                SET enabled = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE task_id IN (
                    SELECT id FROM agent_tasks
                    WHERE assigned_agent_id = ?
                )
                """,
                (agent_id,),
            )
            conn.execute("DELETE FROM agent_memories WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            conn.commit()
        return {"id": agent_id, "archived": False}

    def add_memory(
        self,
        *,
        agent_id: str,
        content: str,
        source_run_id: str | None = None,
        pinned: bool = False,
        source_event_type: str = "manual",
    ) -> dict[str, Any] | None:
        agent = self.get_agent(agent_id)
        if agent is None:
            return None
        if agent.get("is_pseudo"):
            raise PseudoAgentProtectedError("pseudo_agent_protected")

        memory_id = _new_id("mem")
        with get_db_connection(use_row_factory=True) as conn:
            conn.execute(
                """
                INSERT INTO agent_memories (
                    id, agent_id, content, source_run_id, source_event_type, pinned
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    agent_id,
                    content,
                    source_run_id,
                    source_event_type or "manual",
                    1 if pinned else 0,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def list_memories(
        self,
        agent_id: str,
        *,
        include_pinned: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]] | None:
        agent = self.get_agent(agent_id)
        if agent is None:
            return None
        if agent.get("is_pseudo"):
            return []

        clauses = ["agent_id = ?"]
        values: list[Any] = [agent_id]
        if not include_pinned:
            clauses.append("pinned = 0")
        values.append(max(1, min(int(limit), 500)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM agent_memories
                WHERE {' AND '.join(clauses)}
                ORDER BY pinned DESC, created_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def count_memories(self, agent_id: str) -> int | None:
        agent = self.get_agent(agent_id)
        if agent is None:
            return None
        if agent.get("is_pseudo"):
            return 0
        with get_db_connection() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM agent_memories WHERE agent_id = ?",
                    (agent_id,),
                ).fetchone()[0]
            )

    def update_memory(
        self,
        agent_id: str,
        memory_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        agent = self.get_agent(agent_id)
        if agent is None:
            return None
        if agent.get("is_pseudo"):
            raise PseudoAgentProtectedError("pseudo_agent_protected")

        assignments: list[str] = []
        values: list[Any] = []
        if "content" in updates:
            assignments.append("content = ?")
            values.append(updates["content"])
        if "pinned" in updates:
            assignments.append("pinned = ?")
            values.append(1 if updates["pinned"] else 0)

        with get_db_connection(use_row_factory=True) as conn:
            if not assignments:
                row = conn.execute(
                    """
                    SELECT * FROM agent_memories
                    WHERE id = ? AND agent_id = ?
                    """,
                    (memory_id, agent_id),
                ).fetchone()
                return _row_to_memory(row) if row else None
            values.extend([memory_id, agent_id])
            cursor = conn.execute(
                f"""
                UPDATE agent_memories
                SET {', '.join(assignments)}
                WHERE id = ? AND agent_id = ?
                """,
                values,
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                """
                SELECT * FROM agent_memories
                WHERE id = ? AND agent_id = ?
                """,
                (memory_id, agent_id),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def delete_memory(self, agent_id: str, memory_id: str) -> bool | None:
        agent = self.get_agent(agent_id)
        if agent is None:
            return None
        if agent.get("is_pseudo"):
            raise PseudoAgentProtectedError("pseudo_agent_protected")

        with get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM agent_memories WHERE id = ? AND agent_id = ?",
                (memory_id, agent_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def create_run(
        self,
        *,
        project_name: str | None = None,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
        title: str | None = None,
        goal: str | None = None,
        cwd: str | None = None,
        parent_run_id: str | None = None,
        native_session_id: str | None = None,
        task_id: str | None = None,
        dry_run: bool = False,
        simulated_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = _new_id("run")
        status = "queued"
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    id, workspace_id, project_name, task_id, agent_id, provider_id, model, status,
                    title, goal, cwd, native_session_id, parent_run_id, dry_run, simulated_summary_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    workspace_id,
                    project_name,
                    task_id,
                    agent_id,
                    provider_id,
                    model,
                    status,
                    title,
                    goal,
                    cwd,
                    native_session_id,
                    parent_run_id,
                    1 if dry_run else 0,
                    _json_dumps(simulated_summary) if simulated_summary is not None else None,
                ),
            )
            conn.commit()
        if task_id:
            self.link_task_run(task_id=task_id, run_id=run_id, role="primary")
        return self.get_run(run_id) or {}

    def update_run_simulated_summary(
        self,
        run_id: str,
        summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE agent_runs
                SET simulated_summary_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (_json_dumps(summary), run_id),
            )
            conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_run(run_id)

    def simulate_run_from_workflow(
        self,
        *,
        agent: dict[str, Any],
        task: dict[str, Any] | None,
        run_id: str,
    ) -> dict[str, Any]:
        """Append descriptive dry-run events for each configured workflow step."""
        workflow = agent.get("flow_json")
        if not isinstance(workflow, list):
            workflow = []
        if not workflow:
            workflow = [
                {
                    "name": "Preview autonomous run",
                    "description": "No workflow steps are configured; the agent would plan from the task goal.",
                }
            ]

        tools = agent.get("tools_json")
        tool_lookup: dict[str, dict[str, Any]] = {}
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                mcp_id = tool.get("mcp_id")
                if isinstance(mcp_id, str) and mcp_id:
                    tool_lookup[mcp_id] = tool

        task_goal = None
        if isinstance(task, dict):
            task_goal = task.get("goal") or task.get("title")
        simulated = 0
        for index, raw_step in enumerate(workflow, start=1):
            step = raw_step if isinstance(raw_step, dict) else {"name": str(raw_step)}
            step_name = str(step.get("name") or step.get("title") or f"Step {index}")
            description = str(step.get("description") or step.get("success_criteria") or "").strip()
            app_event: dict[str, Any] = {
                "step_index": index,
                "step_name": step_name,
                "would_do": description or f"Would execute workflow step '{step_name}'.",
                "task_goal": task_goal,
            }
            tool_hint = step.get("tool_hint")
            if isinstance(tool_hint, str) and tool_hint.strip():
                tool = tool_lookup.get(tool_hint.strip(), {})
                examples = tool.get("user_examples") or tool.get("tool_names") or []
                app_event["would_call_mcp"] = {
                    "mcp_id": tool_hint.strip(),
                    "examples": list(examples) if isinstance(examples, list) else [],
                }
            self.append_event(
                run_id=run_id,
                event_type="step.simulated",
                app_event=app_event,
            )
            simulated += 1

        summary = {
            "agent_id": agent.get("id"),
            "task_id": task.get("id") if isinstance(task, dict) else None,
            "steps_simulated": simulated,
            # Leads with the fact a reader must not miss: this produced no
            # side effects at all. The previous wording ("Dry run finished —
            # N steps simulated") read step-by-step like a completed run, and
            # a first-time user took "finished" plus a step count as "it
            # worked" — see agent-usability-repair symptom #2.
            "summary": (
                f"Simulation only — nothing was executed. {simulated} steps simulated."
            ),
        }
        self.update_run_simulated_summary(run_id, summary)
        return summary

    def list_runs(
        self,
        *,
        project_name: str | None = None,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
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
        if agent_id:
            clauses.append("agent_id = ?")
            values.append(agent_id)
        if task_id:
            clauses.append("task_id = ?")
            values.append(task_id)
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
        return [_row_to_run_record(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return _row_to_run_record(row) if row else None

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

    def create_task(
        self,
        *,
        title: str,
        description: str | None = None,
        project_name: str | None = None,
        workspace_id: str | None = None,
        run_id: str | None = None,
        assigned_agent_id: str | None = None,
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
                    id, workspace_id, run_id, assigned_agent_id, project_name, kind, source,
                    title, description, goal, priority, due_at, labels_json,
                    assignee, requester, acceptance_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    workspace_id,
                    run_id,
                    assigned_agent_id,
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
            "assigned_agent_id",
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

    def get_task_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """Return the active waiting checkpoint for a task, if any."""
        task = self.get_task(task_id)
        if task is None:
            return None

        for step in self.list_task_steps(task_id):
            if step.get("status") not in _CHECKPOINT_STEP_STATUSES:
                continue
            checkpoint = _checkpoint_from_step(step)
            if checkpoint is None:
                continue
            run = None
            step_run_id = step.get("run_id") or task.get("run_id")
            if isinstance(step_run_id, str) and step_run_id:
                run = self.get_run(step_run_id)
            return {
                "task": task,
                "run": run,
                "step": step,
                "checkpoint": checkpoint,
                "connector_requests": self.list_connector_requests(
                    task_id=task_id,
                    step_id=step["id"],
                    limit=100,
                ),
            }

        return {
            "task": task,
            "run": self.get_run(task["run_id"]) if task.get("run_id") else None,
            "step": None,
            "checkpoint": None,
            "connector_requests": [],
        }

    def get_run_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Return the active waiting checkpoint for a run, if any."""
        run = self.get_run(run_id)
        if run is None:
            return None
        task_id = run.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return {
                "task": None,
                "run": run,
                "step": None,
                "checkpoint": None,
                "connector_requests": [],
            }
        checkpoint = self.get_task_checkpoint(task_id)
        if checkpoint is None:
            return None
        checkpoint["run"] = checkpoint.get("run") or run
        return checkpoint

    def append_step_user_response(
        self,
        *,
        task_id: str,
        step_id: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        remember: bool = False,
    ) -> dict[str, Any] | None:
        """Append a user response to a waiting step checkpoint.

        The first runtime iteration keeps user intervention history in the
        step output JSON. That avoids a migration while still making the
        response durable and timeline-visible.
        """
        task = self.get_task(task_id)
        step = self.get_task_step(step_id)
        if task is None or step is None or step.get("task_id") != task_id:
            return None
        if step.get("status") not in _CHECKPOINT_STEP_STATUSES:
            raise AgentStoreConflictError("step_not_waiting_for_user")

        content = str(message or "").strip()
        if not content:
            raise ValueError("message is required")

        response = {
            "message": content,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        }
        output = dict(step.get("output") or {})
        responses = output.get("user_responses")
        if not isinstance(responses, list):
            responses = []
        responses.append(response)
        output["user_responses"] = responses
        output["last_user_response"] = response
        updated_step = self.update_task_step(step_id, {"output": output})

        run_id = step.get("run_id") or task.get("run_id")
        if isinstance(run_id, str) and run_id:
            self.append_event(
                run_id=run_id,
                event_type="task.step.user_response",
                app_event={
                    "task_id": task_id,
                    "step_id": step_id,
                    "response": response,
                },
            )

        memory = None
        agent_id = task.get("assigned_agent_id")
        if remember and isinstance(agent_id, str) and agent_id:
            memory = self.add_memory(
                agent_id=agent_id,
                content=content,
                source_run_id=run_id if isinstance(run_id, str) else None,
                source_event_type="user_response",
            )

        return {
            "task": self.get_task(task_id),
            "step": updated_step,
            "response": response,
            "memory": memory,
        }

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


def add_memory_from_event(
    *,
    agent_id: str,
    content: str,
    source_run_id: str,
    source_event_type: str = "user_annotation",
    pinned: bool = False,
) -> dict[str, Any] | None:
    """Persist a run annotation as agent memory unless the agent is pseudo."""
    store = get_agent_store()
    agent = store.get_agent(agent_id)
    if agent is None:
        return None
    if agent.get("is_pseudo"):
        logger.warning(
            "skipping memory annotation for pseudo-agent agent_id=%s source_run_id=%s",
            agent_id,
            source_run_id,
        )
        return None
    return store.add_memory(
        agent_id=agent_id,
        content=content,
        source_run_id=source_run_id,
        source_event_type=source_event_type,
        pinned=pinned,
    )
