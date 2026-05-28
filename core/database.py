"""SQLite database management for Code Bridge."""

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from core.runtime_paths import SERVER_DIR, runtime_path

logger = logging.getLogger(__name__)

DB_PATH = runtime_path("code_bridge.db", SERVER_DIR / "core" / "code_bridge.db")

AGENT_COCKPIT_SCHEMA_VERSION = 2026052801
WORK_COCKPIT_SCHEMA_VERSION = 2026052802
TASK_SCHEDULES_SCHEMA_VERSION = 2026052803


@contextmanager
def get_db_connection(
    use_row_factory: bool = False,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections.

    Args:
        use_row_factory: If True, sets row_factory to sqlite3.Row for dict-like access.

    Yields:
        SQLite connection that auto-closes on exit.

    Example:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute("SELECT * FROM projects").fetchall()
    """
    conn = sqlite3.connect(DB_PATH)
    if use_row_factory:
        conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _applied_schema_versions(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {int(row[0]) for row in rows}


def _record_schema_migration(
    conn: sqlite3.Connection,
    *,
    version: int,
    name: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (version, name)
        VALUES (?, ?)
        """,
        (version, name),
    )


def _migrate_agent_cockpit_foundation(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            project_name TEXT,
            type TEXT NOT NULL DEFAULT 'code_project',
            root_path TEXT NOT NULL,
            display_name TEXT,
            permissions_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_workspaces_project_name
        ON workspaces(project_name);

        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            project_name TEXT,
            provider_id TEXT,
            model TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            title TEXT,
            goal TEXT,
            cwd TEXT,
            native_session_id TEXT,
            parent_run_id TEXT,
            started_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_runs_workspace_updated
        ON agent_runs(workspace_id, updated_at);

        CREATE INDEX IF NOT EXISTS idx_agent_runs_project_updated
        ON agent_runs(project_name, updated_at);

        CREATE INDEX IF NOT EXISTS idx_agent_runs_status
        ON agent_runs(status);

        CREATE TABLE IF NOT EXISTS agent_tasks (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            run_id TEXT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            priority INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_tasks_workspace_status
        ON agent_tasks(workspace_id, status, updated_at);

        CREATE INDEX IF NOT EXISTS idx_agent_tasks_run_id
        ON agent_tasks(run_id);

        CREATE TABLE IF NOT EXISTS agent_events (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            provider_id TEXT,
            provider_event_json TEXT,
            app_event_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, sequence)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_events_run_sequence
        ON agent_events(run_id, sequence);

        CREATE INDEX IF NOT EXISTS idx_agent_events_created_at
        ON agent_events(created_at);

        CREATE TABLE IF NOT EXISTS agent_messages (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            attachments_json TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_messages_run_created
        ON agent_messages(run_id, created_at);

        CREATE TABLE IF NOT EXISTS agent_permissions (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            request_id TEXT,
            tool_name TEXT NOT NULL,
            input_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_permissions_run_status
        ON agent_permissions(run_id, status, created_at);

        CREATE INDEX IF NOT EXISTS idx_agent_permissions_request
        ON agent_permissions(request_id);

        CREATE TABLE IF NOT EXISTS agent_artifacts (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            kind TEXT NOT NULL,
            path TEXT,
            mime_type TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_artifacts_run_kind
        ON agent_artifacts(run_id, kind, created_at);

        CREATE TABLE IF NOT EXISTS approval_requests (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            actor_json TEXT NOT NULL DEFAULT '{}',
            operation TEXT NOT NULL,
            risk_level TEXT NOT NULL DEFAULT 'medium',
            details_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            resolved_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_approval_requests_status
        ON approval_requests(status, created_at);

        CREATE INDEX IF NOT EXISTS idx_approval_requests_run_id
        ON approval_requests(run_id);

        CREATE TABLE IF NOT EXISTS approval_decisions (
            id TEXT PRIMARY KEY,
            approval_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'once',
            reason TEXT,
            constraints_json TEXT NOT NULL DEFAULT '{}',
            approver_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_approval_decisions_approval
        ON approval_decisions(approval_id, created_at);

        CREATE TABLE IF NOT EXISTS policy_rules (
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            operation TEXT NOT NULL,
            effect TEXT NOT NULL,
            constraints_json TEXT NOT NULL DEFAULT '{}',
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_policy_rules_scope_operation
        ON policy_rules(scope, operation);

        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            actor_type TEXT,
            client_id TEXT,
            device_name TEXT,
            project_name TEXT,
            provider_id TEXT,
            model TEXT,
            session_id TEXT,
            run_id TEXT,
            operation TEXT NOT NULL,
            resource TEXT,
            risk_level TEXT,
            decision TEXT,
            payload_redacted_json TEXT NOT NULL DEFAULT '{}',
            affected_paths_json TEXT NOT NULL DEFAULT '[]',
            hash_prev TEXT,
            hash_current TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
        ON audit_events(timestamp);

        CREATE INDEX IF NOT EXISTS idx_audit_events_run_id
        ON audit_events(run_id);

        CREATE INDEX IF NOT EXISTS idx_audit_events_project
        ON audit_events(project_name, timestamp);

        CREATE TABLE IF NOT EXISTS preview_sessions (
            id TEXT PRIMARY KEY,
            project_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            port INTEGER,
            url TEXT,
            device_id TEXT,
            emulator_profile_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_preview_sessions_project
        ON preview_sessions(project_name, updated_at);
    """)


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row[1]) for row in rows}
    if column in existing:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_work_cockpit_foundation(conn: sqlite3.Connection) -> None:
    for column, definition in (
        ("project_name", "TEXT"),
        ("kind", "TEXT NOT NULL DEFAULT 'general'"),
        ("source", "TEXT NOT NULL DEFAULT 'manual'"),
        ("goal", "TEXT"),
        ("due_at", "TIMESTAMP"),
        ("labels_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("assignee", "TEXT"),
        ("requester", "TEXT"),
        ("acceptance_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("metadata_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("result_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("error_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("started_at", "TIMESTAMP"),
        ("ended_at", "TIMESTAMP"),
    ):
        _add_column_if_missing(conn, "agent_tasks", column, definition)

    _add_column_if_missing(conn, "agent_runs", "task_id", "TEXT")

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_project_status
        ON agent_tasks(project_name, status, updated_at);

        CREATE INDEX IF NOT EXISTS idx_agent_tasks_kind_status
        ON agent_tasks(kind, status, updated_at);

        CREATE INDEX IF NOT EXISTS idx_agent_runs_task_id
        ON agent_runs(task_id);

        CREATE TABLE IF NOT EXISTS agent_task_runs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'primary',
            sequence INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(task_id, run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_task_runs_task_sequence
        ON agent_task_runs(task_id, sequence, created_at);

        CREATE INDEX IF NOT EXISTS idx_agent_task_runs_run_id
        ON agent_task_runs(run_id);

        CREATE TABLE IF NOT EXISTS agent_task_steps (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            run_id TEXT,
            capability_id TEXT,
            sequence INTEGER NOT NULL,
            title TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            input_json TEXT NOT NULL DEFAULT '{}',
            output_json TEXT NOT NULL DEFAULT '{}',
            approval_id TEXT,
            artifact_id TEXT,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(task_id, sequence)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_task_steps_task_sequence
        ON agent_task_steps(task_id, sequence);

        CREATE TABLE IF NOT EXISTS agent_capabilities (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            provider_id TEXT,
            status TEXT NOT NULL DEFAULT 'available',
            scope TEXT NOT NULL DEFAULT 'global',
            source TEXT NOT NULL DEFAULT 'codebridge',
            description TEXT,
            permission_level TEXT NOT NULL DEFAULT 'approval',
            desktop_only INTEGER NOT NULL DEFAULT 0,
            local_only INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_capabilities_type_status
        ON agent_capabilities(type, status, updated_at);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_capabilities_identity
        ON agent_capabilities(type, name, IFNULL(provider_id, ''), source);

        CREATE TABLE IF NOT EXISTS agent_task_capabilities (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            capability_id TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'read',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(task_id, capability_id, mode)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_task_capabilities_task
        ON agent_task_capabilities(task_id);

        CREATE TABLE IF NOT EXISTS agent_connector_requests (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            step_id TEXT,
            run_id TEXT,
            connector_type TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending_review',
            adapter_json TEXT NOT NULL DEFAULT '{}',
            parameters_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            error_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_connector_requests_task_status
        ON agent_connector_requests(task_id, status, updated_at);

        CREATE INDEX IF NOT EXISTS idx_agent_connector_requests_step
        ON agent_connector_requests(step_id);

        CREATE TABLE IF NOT EXISTS usage_events (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'chat',
            project_name TEXT,
            workspace_id TEXT,
            task_id TEXT,
            run_id TEXT,
            provider_id TEXT,
            model TEXT,
            native_session_id TEXT,
            turn_id TEXT,
            duration_ms INTEGER,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0,
            raw_usage_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_usage_events_created_at
        ON usage_events(created_at);

        CREATE INDEX IF NOT EXISTS idx_usage_events_project_created
        ON usage_events(project_name, created_at);

        CREATE INDEX IF NOT EXISTS idx_usage_events_task_created
        ON usage_events(task_id, created_at);

        CREATE INDEX IF NOT EXISTS idx_usage_events_run_created
        ON usage_events(run_id, created_at);
    """)


def _migrate_task_schedules(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_schedules (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            name TEXT,
            expression_json TEXT NOT NULL,
            provider_id TEXT,
            model TEXT,
            cwd TEXT,
            prompt TEXT,
            capabilities_json TEXT NOT NULL DEFAULT '[]',
            enabled INTEGER NOT NULL DEFAULT 1,
            skip_if_active INTEGER NOT NULL DEFAULT 1,
            next_run_at TIMESTAMP,
            last_run_at TIMESTAMP,
            last_run_id TEXT,
            last_status TEXT,
            last_error TEXT,
            fire_count INTEGER NOT NULL DEFAULT 0,
            skip_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_task_schedules_task
        ON task_schedules(task_id);

        CREATE INDEX IF NOT EXISTS idx_task_schedules_next_run
        ON task_schedules(enabled, next_run_at);
    """)


_SCHEMA_MIGRATIONS: tuple[tuple[int, str, Callable[[sqlite3.Connection], None]], ...] = (
    (
        AGENT_COCKPIT_SCHEMA_VERSION,
        "agent_cockpit_foundation",
        _migrate_agent_cockpit_foundation,
    ),
    (
        WORK_COCKPIT_SCHEMA_VERSION,
        "work_cockpit_foundation",
        _migrate_work_cockpit_foundation,
    ),
    (
        TASK_SCHEDULES_SCHEMA_VERSION,
        "task_schedules",
        _migrate_task_schedules,
    ),
)


def _run_schema_migrations(conn: sqlite3.Connection) -> None:
    _ensure_schema_migrations_table(conn)
    applied_versions = _applied_schema_versions(conn)
    for version, name, migration in _SCHEMA_MIGRATIONS:
        if version in applied_versions:
            continue
        migration(conn)
        _record_schema_migration(conn, version=version, name=name)
        logger.info("Applied schema migration %s (%s)", version, name)


def init_db() -> None:
    """Initialize database and create tables."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                type TEXT DEFAULT 'flutter',
                dev_server_command TEXT,
                dev_server_port INTEGER,
                enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: add enabled column if missing
        try:
            conn.execute("ALTER TABLE projects ADD COLUMN enabled INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                cost_usd REAL NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_turns_created_at
            ON usage_turns(created_at)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accessible_folders (
                path TEXT PRIMARY KEY,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _run_schema_migrations(conn)
        conn.commit()


class ProjectDB:
    """Database operations for projects."""

    def __init__(self):
        init_db()

    def get_all(self) -> list[dict]:
        """Get all projects."""
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get(self, name: str) -> Optional[dict]:
        """Get a project by name."""
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE name = ?", (name,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def create(self, project: dict) -> dict:
        """Create a new project."""
        dev_server = project.get("dev_server") or {}
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO projects (name, path, type, dev_server_command, dev_server_port)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    project["name"],
                    project["path"],
                    project.get("type", "flutter"),
                    dev_server.get("command"),
                    dev_server.get("port"),
                ),
            )
            conn.commit()
        return self.get(project["name"])

    def update(self, name: str, data: dict) -> Optional[dict]:
        """Update an existing project."""
        # Build dynamic UPDATE query
        updates = []
        values = []

        if "path" in data:
            updates.append("path = ?")
            values.append(data["path"])

        if "type" in data:
            updates.append("type = ?")
            values.append(data["type"])

        if "dev_server" in data and data["dev_server"] is not None:
            dev_server = data["dev_server"]
            if "command" in dev_server:
                updates.append("dev_server_command = ?")
                values.append(dev_server["command"])
            if "port" in dev_server:
                updates.append("dev_server_port = ?")
                values.append(dev_server["port"])

        if "enabled" in data:
            updates.append("enabled = ?")
            values.append(1 if data["enabled"] else 0)

        if not updates:
            return self.get(name)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(name)

        query = f"UPDATE projects SET {', '.join(updates)} WHERE name = ?"
        with get_db_connection() as conn:
            conn.execute(query, values)
            conn.commit()

        return self.get(name)

    def delete(self, name: str) -> bool:
        """Delete a project."""
        with get_db_connection() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0

    def exists(self, name: str) -> bool:
        """Check if a project exists."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM projects WHERE name = ?", (name,)
            ).fetchone()
            return row is not None

    def migrate_from_config(self, projects: list[dict]) -> int:
        """Migrate projects from config.yaml to database."""
        count = 0
        for project in projects:
            if not self.exists(project.get("name", "")):
                self.create(project)
                count += 1
        return count

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert database row to project dict."""
        return {
            "name": row["name"],
            "path": row["path"],
            "type": row["type"],
            "dev_server": (
                {
                    "command": row["dev_server_command"],
                    "port": row["dev_server_port"],
                }
                if (row["dev_server_command"] or row["dev_server_port"])
                else None
            ),
            "enabled": bool(row["enabled"]) if row["enabled"] is not None else True,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


class UsageDB:
    """Database operations for usage/cost metrics."""

    def __init__(self):
        init_db()

    def record_turn(
        self,
        project_name: str,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Persist one completed turn usage row."""
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_turns (project_name, cost_usd, input_tokens, output_tokens)
                VALUES (?, ?, ?, ?)
            """,
                (
                    project_name,
                    max(float(cost_usd), 0.0),
                    max(int(input_tokens), 0),
                    max(int(output_tokens), 0),
                ),
            )
            conn.commit()

    def record_event(
        self,
        *,
        source: str = "chat",
        project_name: str | None = None,
        workspace_id: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
        native_session_id: str | None = None,
        turn_id: str | None = None,
        duration_ms: int | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
        raw_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one task/run-aware usage event."""
        event_id = f"use_{uuid.uuid4().hex}"
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_events (
                    id, source, project_name, workspace_id, task_id, run_id,
                    provider_id, model, native_session_id, turn_id, duration_ms,
                    input_tokens, output_tokens, cache_tokens, reasoning_tokens,
                    cost_usd, raw_usage_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    source,
                    project_name,
                    workspace_id,
                    task_id,
                    run_id,
                    provider_id,
                    model,
                    native_session_id,
                    turn_id,
                    int(duration_ms) if duration_ms is not None else None,
                    max(int(input_tokens), 0),
                    max(int(output_tokens), 0),
                    max(int(cache_tokens), 0),
                    max(int(reasoning_tokens), 0),
                    max(float(cost_usd), 0.0),
                    json.dumps(raw_usage or {}, separators=(",", ":"), ensure_ascii=False),
                ),
            )
            conn.commit()
        return self.get_event(event_id) or {}

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM usage_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_usage_event(row) if row else None

    def get_weekly_summary(
        self,
        budget_usd: float | None = None,
        window_days: int = 7,
        *,
        project_name: str | None = None,
        workspace_id: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        provider_id: str | None = None,
    ) -> dict:
        """Return rolling-window usage summary and optional budget percentage."""
        filters, values = self._usage_filters(
            window_days=window_days,
            project_name=project_name,
            workspace_id=workspace_id,
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
        )
        where_sql = f"WHERE {' AND '.join(filters)}"
        with get_db_connection(use_row_factory=True) as conn:
            if self._has_usage_events(conn):
                row = conn.execute(
                    f"""
                    SELECT
                        COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                        COUNT(*) AS turn_count,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(cache_tokens), 0) AS cache_tokens,
                        COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                        MAX(created_at) AS last_turn_at
                    FROM usage_events
                    {where_sql}
                    """,
                    values,
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT
                        COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                        COUNT(*) AS turn_count,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        0 AS cache_tokens,
                        0 AS reasoning_tokens,
                        MAX(created_at) AS last_turn_at
                    FROM usage_turns
                    WHERE created_at >= datetime('now', ?)
                    """,
                    (f"-{max(window_days, 1)} days",),
                ).fetchone()
            total_cost = float(row["total_cost_usd"] or 0.0)
            turn_count = int(row["turn_count"] or 0)
            total_input_tokens = int(row["input_tokens"] or 0)
            total_output_tokens = int(row["output_tokens"] or 0)
            total_cache_tokens = int(row["cache_tokens"] or 0)
            total_reasoning_tokens = int(row["reasoning_tokens"] or 0)
            last_turn_at = row["last_turn_at"]

        has_budget = budget_usd is not None and float(budget_usd) > 0
        usage_percent = None
        if has_budget:
            usage_percent = (total_cost / float(budget_usd)) * 100.0

        return {
            "window_days": int(max(window_days, 1)),
            "total_cost_usd": round(total_cost, 6),
            "turn_count": turn_count,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_tokens": total_cache_tokens,
            "reasoning_tokens": total_reasoning_tokens,
            "budget_usd": float(budget_usd) if has_budget else None,
            "has_budget": has_budget,
            "usage_percent": round(usage_percent, 2) if usage_percent is not None else None,
            "last_turn_at": last_turn_at,
        }

    def list_events(
        self,
        *,
        window_days: int = 7,
        project_name: str | None = None,
        workspace_id: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        provider_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        filters, values = self._usage_filters(
            window_days=window_days,
            project_name=project_name,
            workspace_id=workspace_id,
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
        )
        values.append(max(1, min(int(limit), 500)))
        with get_db_connection(use_row_factory=True) as conn:
            if not self._has_usage_events(conn):
                return []
            rows = conn.execute(
                f"""
                SELECT * FROM usage_events
                WHERE {' AND '.join(filters)}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [self._row_to_usage_event(row) for row in rows]

    def breakdown(
        self,
        group_by: str,
        *,
        window_days: int = 7,
        project_name: str | None = None,
        workspace_id: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        provider_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        allowed = {
            "project_name",
            "workspace_id",
            "task_id",
            "run_id",
            "provider_id",
            "model",
            "source",
        }
        resolved_group = group_by if group_by in allowed else "provider_id"
        filters, values = self._usage_filters(
            window_days=window_days,
            project_name=project_name,
            workspace_id=workspace_id,
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
        )
        values.append(max(1, min(int(limit), 100)))
        with get_db_connection(use_row_factory=True) as conn:
            if not self._has_usage_events(conn):
                return []
            rows = conn.execute(
                f"""
                SELECT
                    COALESCE({resolved_group}, 'unassigned') AS key,
                    COUNT(*) AS turn_count,
                    COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cache_tokens), 0) AS cache_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    MAX(created_at) AS last_turn_at
                FROM usage_events
                WHERE {' AND '.join(filters)}
                GROUP BY COALESCE({resolved_group}, 'unassigned')
                ORDER BY total_cost_usd DESC, turn_count DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [
            {
                "key": row["key"],
                "turn_count": int(row["turn_count"] or 0),
                "total_cost_usd": round(float(row["total_cost_usd"] or 0), 6),
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "cache_tokens": int(row["cache_tokens"] or 0),
                "reasoning_tokens": int(row["reasoning_tokens"] or 0),
                "last_turn_at": row["last_turn_at"],
            }
            for row in rows
        ]

    def _usage_filters(
        self,
        *,
        window_days: int,
        project_name: str | None = None,
        workspace_id: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        provider_id: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        filters = ["created_at >= datetime('now', ?)"]
        values: list[Any] = [f"-{max(window_days, 1)} days"]
        for column, value in (
            ("project_name", project_name),
            ("workspace_id", workspace_id),
            ("task_id", task_id),
            ("run_id", run_id),
            ("provider_id", provider_id),
        ):
            if value:
                filters.append(f"{column} = ?")
                values.append(value)
        return filters, values

    def _has_usage_events(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'usage_events'"
        ).fetchone()
        return row is not None

    def _row_to_usage_event(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            raw_usage = json.loads(row["raw_usage_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_usage = {}
        return {
            "id": row["id"],
            "source": row["source"],
            "project_name": row["project_name"],
            "workspace_id": row["workspace_id"],
            "task_id": row["task_id"],
            "run_id": row["run_id"],
            "provider_id": row["provider_id"],
            "model": row["model"],
            "native_session_id": row["native_session_id"],
            "turn_id": row["turn_id"],
            "duration_ms": row["duration_ms"],
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "cache_tokens": int(row["cache_tokens"] or 0),
            "reasoning_tokens": int(row["reasoning_tokens"] or 0),
            "cost_usd": round(float(row["cost_usd"] or 0), 6),
            "raw_usage": raw_usage,
            "created_at": row["created_at"],
        }


class SettingsDB:
    """Database operations for lightweight key-value app settings."""

    def __init__(self):
        init_db()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get one setting value by key."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return default
            return row[0]

    def set(self, key: str, value: Optional[str]) -> None:
        """Upsert one setting value."""
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (key, value),
            )
            conn.commit()

    def get_json(self, key: str, default: Any) -> Any:
        """Get one JSON setting value by key."""
        raw_value = self.get(key)
        if raw_value is None:
            return default
        try:
            return json.loads(raw_value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return default

    def set_json(self, key: str, value: Any) -> None:
        """Store one JSON-serializable value."""
        self.set(key, json.dumps(value))


class AccessibleFolderDB:
    """Database operations for accessible folders (security boundary).

    These folders define what paths the server can access.
    This is SEPARATE from the projects list.
    """

    def __init__(self):
        init_db()

    def get_all(self) -> list[str]:
        """Get all accessible folder paths."""
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT path FROM accessible_folders ORDER BY path"
            ).fetchall()
            return [row[0] for row in rows]

    def add(self, path: str) -> bool:
        """Add an accessible folder path.

        Returns True if added, False if already exists.
        """
        resolved_path = str(Path(path).expanduser().resolve())

        with get_db_connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO accessible_folders (path) VALUES (?)",
                    (resolved_path,),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def remove(self, path: str) -> bool:
        """Remove an accessible folder path.

        Returns True if removed, False if not found.
        """
        resolved_path = str(Path(path).expanduser().resolve())

        with get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM accessible_folders WHERE path = ?",
                (resolved_path,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def exists(self, path: str) -> bool:
        """Check if a path is in accessible folders."""
        resolved_path = str(Path(path).expanduser().resolve())

        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM accessible_folders WHERE path = ?",
                (resolved_path,),
            ).fetchone()
            return row is not None


# Global database instance
_project_db: ProjectDB | None = None
_usage_db: UsageDB | None = None
_settings_db: SettingsDB | None = None
_accessible_folder_db: AccessibleFolderDB | None = None


def get_project_db() -> ProjectDB:
    """Get global project database instance."""
    global _project_db
    if _project_db is None:
        _project_db = ProjectDB()
    return _project_db


def get_usage_db() -> UsageDB:
    """Get global usage database instance."""
    global _usage_db
    if _usage_db is None:
        _usage_db = UsageDB()
    return _usage_db


def get_settings_db() -> SettingsDB:
    """Get global settings database instance."""
    global _settings_db
    if _settings_db is None:
        _settings_db = SettingsDB()
    return _settings_db


def get_accessible_folder_db() -> AccessibleFolderDB:
    """Get global accessible folder database instance."""
    global _accessible_folder_db
    if _accessible_folder_db is None:
        _accessible_folder_db = AccessibleFolderDB()
    return _accessible_folder_db


def migrate_accessible_folders_from_projects() -> int:
    """One-time migration: seed accessible_folders from existing projects.

    Extracts unique parent directories from all registered projects
    and adds them to the accessible_folders table.
    If no projects exist, adds the home directory as default.

    Returns:
        Number of folders migrated.
    """
    project_db = get_project_db()
    folder_db = get_accessible_folder_db()

    # Skip if already has folders
    existing_folders = folder_db.get_all()
    if existing_folders:
        return 0

    projects = project_db.get_all()

    # No projects - add home directory as default
    if not projects:
        home_dir = str(Path.home())
        if folder_db.add(home_dir):
            logger.info("Added home directory as default accessible folder: %s", home_dir)
            return 1
        return 0

    # Extract unique parent folders
    parent_folders: set[str] = set()
    for proj in projects:
        project_path = proj.get("path")
        if project_path:
            parent = str(Path(project_path).parent)
            parent_folders.add(parent)

    # Add to accessible_folders
    count = 0
    for folder in parent_folders:
        if folder_db.add(folder):
            count += 1

    if count > 0:
        logger.info("Migrated %d accessible folders from existing projects", count)

    return count
