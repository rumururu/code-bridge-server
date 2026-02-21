"""SQLite database management for Code Bridge."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).parent / "code_bridge.db"


def init_db():
    """Initialize database and create tables."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            name TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            type TEXT DEFAULT 'flutter',
            dev_server_command TEXT,
            dev_server_port INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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
    conn.commit()
    conn.close()


class ProjectDB:
    """Database operations for projects."""

    def __init__(self):
        init_db()

    def get_all(self) -> list[dict]:
        """Get all projects."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
        conn.close()
        return [self._row_to_dict(row) for row in rows]

    def get(self, name: str) -> Optional[dict]:
        """Get a project by name."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM projects WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
        return self._row_to_dict(row) if row else None

    def create(self, project: dict) -> dict:
        """Create a new project."""
        conn = sqlite3.connect(DB_PATH)
        dev_server = project.get("dev_server") or {}
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
        conn.close()
        return self.get(project["name"])

    def update(self, name: str, data: dict) -> Optional[dict]:
        """Update an existing project."""
        conn = sqlite3.connect(DB_PATH)

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

        if not updates:
            conn.close()
            return self.get(name)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(name)

        query = f"UPDATE projects SET {', '.join(updates)} WHERE name = ?"
        conn.execute(query, values)
        conn.commit()
        conn.close()

        return self.get(name)

    def delete(self, name: str) -> bool:
        """Delete a project."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    def exists(self, name: str) -> bool:
        """Check if a project exists."""
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT 1 FROM projects WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
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
            "dev_server": {
                "command": row["dev_server_command"],
                "port": row["dev_server_port"],
            }
            if row["dev_server_port"]
            else None,
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
        conn = sqlite3.connect(DB_PATH)
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
        conn.close()

    def get_weekly_summary(self, budget_usd: float | None = None, window_days: int = 7) -> dict:
        """Return rolling-window usage summary and optional budget percentage."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                COUNT(*) AS turn_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                MAX(created_at) AS last_turn_at
            FROM usage_turns
            WHERE created_at >= datetime('now', ?)
        """,
            (f"-{max(window_days, 1)} days",),
        ).fetchone()
        conn.close()

        total_cost = float(row["total_cost_usd"] or 0.0)
        turn_count = int(row["turn_count"] or 0)
        total_input_tokens = int(row["input_tokens"] or 0)
        total_output_tokens = int(row["output_tokens"] or 0)
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
            "budget_usd": float(budget_usd) if has_budget else None,
            "has_budget": has_budget,
            "usage_percent": round(usage_percent, 2) if usage_percent is not None else None,
            "last_turn_at": last_turn_at,
        }


class SettingsDB:
    """Database operations for lightweight key-value app settings."""

    def __init__(self):
        init_db()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get one setting value by key."""
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            (key,),
        ).fetchone()
        conn.close()
        if row is None:
            return default
        return row[0]

    def set(self, key: str, value: Optional[str]) -> None:
        """Upsert one setting value."""
        conn = sqlite3.connect(DB_PATH)
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
        conn.close()

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


# Global database instance
_project_db: ProjectDB | None = None
_usage_db: UsageDB | None = None
_settings_db: SettingsDB | None = None


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
