"""SQLite persistence for Agent Cockpit workspaces."""

import json
import uuid
from typing import Any

from core.database import get_db_connection, init_db


def _new_id() -> str:
    return f"wsp_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _row_to_workspace(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project_name": row["project_name"],
        "type": row["type"],
        "root_path": row["root_path"],
        "display_name": row["display_name"],
        "permissions": _json_loads(row["permissions_json"], {}),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class WorkspaceStore:
    """Persistence helper for workspace records."""

    def __init__(self) -> None:
        init_db()

    def create_workspace(
        self,
        *,
        root_path: str,
        project_name: str | None = None,
        type: str = "code_project",
        display_name: str | None = None,
        permissions: dict[str, Any] | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        workspace_id = _new_id()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO workspaces (
                    id, project_name, type, root_path, display_name,
                    permissions_json, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    project_name,
                    type,
                    root_path,
                    display_name,
                    _json_dumps(permissions or {}),
                    status,
                ),
            )
            conn.commit()
        return self.get_workspace(workspace_id) or {}

    def get_or_create_project_workspace(
        self,
        *,
        project_name: str,
        root_path: str,
        display_name: str | None = None,
        permissions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the active workspace for a project/root pair, creating it if absent."""
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                """
                SELECT * FROM workspaces
                WHERE project_name = ?
                  AND root_path = ?
                  AND status = 'active'
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (project_name, root_path),
            ).fetchone()
        if row:
            return _row_to_workspace(row)
        return self.create_workspace(
            project_name=project_name,
            root_path=root_path,
            display_name=display_name or project_name,
            permissions=permissions or {"roots": [root_path]},
        )

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        return _row_to_workspace(row) if row else None

    def list_workspaces(
        self,
        *,
        project_name: str | None = None,
        status: str | None = "active",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if project_name:
            clauses.append("project_name = ?")
            values.append(project_name)
        if status:
            clauses.append("status = ?")
            values.append(status)
        values.append(max(1, min(int(limit), 500)))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM workspaces
                {where_sql}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_workspace(row) for row in rows]

    def update_workspace(
        self,
        workspace_id: str,
        *,
        project_name: str | None = None,
        type: str | None = None,
        root_path: str | None = None,
        display_name: str | None = None,
        permissions: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        updates: list[str] = []
        values: list[Any] = []
        if project_name is not None:
            updates.append("project_name = ?")
            values.append(project_name)
        if type is not None:
            updates.append("type = ?")
            values.append(type)
        if root_path is not None:
            updates.append("root_path = ?")
            values.append(root_path)
        if display_name is not None:
            updates.append("display_name = ?")
            values.append(display_name)
        if permissions is not None:
            updates.append("permissions_json = ?")
            values.append(_json_dumps(permissions))
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if not updates:
            return self.get_workspace(workspace_id)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(workspace_id)
        with get_db_connection() as conn:
            cursor = conn.execute(
                f"UPDATE workspaces SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_workspace(workspace_id)

    def archive_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        return self.update_workspace(workspace_id, status="archived")


_workspace_store: WorkspaceStore | None = None


def get_workspace_store() -> WorkspaceStore:
    """Return the process-global workspace store."""
    global _workspace_store
    if _workspace_store is None:
        _workspace_store = WorkspaceStore()
    return _workspace_store
