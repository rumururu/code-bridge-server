"""SQLite database management for Code Bridge."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "code_bridge.db"


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

    def get_weekly_summary(self, budget_usd: float | None = None, window_days: int = 7) -> dict:
        """Return rolling-window usage summary and optional budget percentage."""
        with get_db_connection(use_row_factory=True) as conn:
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
