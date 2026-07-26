"""Registry of shell scripts workflow steps are allowed to run.

A ``shell`` step names a registered script by id. It never carries a command
line, because a workflow that could is remote code execution wearing a
workflow's clothes — especially once a standing policy rule lets it run
unattended. Registering is the deliberate, auditable act; running is a lookup.

Registration validates that the path exists and is a regular file, so a typo
fails at registration time instead of at 3am inside a scheduled run.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from core.database import get_db_connection

_DEFAULT_TIMEOUT_SECONDS = 3600
_MAX_TIMEOUT_SECONDS = 6 * 3600
ALLOWED_INTERPRETERS = {"bash", "sh", "zsh", "python3", "node", "direct"}


class ScriptRegistrationError(ValueError):
    """Raised when a script cannot be registered as described."""


def _new_id() -> str:
    return f"script_{uuid.uuid4().hex}"


def _row_to_script(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "path": row["path"],
        "interpreter": row["interpreter"],
        "default_args": json.loads(row["default_args_json"] or "[]"),
        "timeout_seconds": row["timeout_seconds"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _normalize_path(raw_path: str) -> str:
    text = str(raw_path or "").strip()
    if not text:
        raise ScriptRegistrationError("script path is required")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ScriptRegistrationError("script path must be absolute")
    if not path.exists():
        raise ScriptRegistrationError(f"script not found: {path}")
    if not path.is_file():
        raise ScriptRegistrationError(f"script path is not a file: {path}")
    return str(path)


def _normalize_interpreter(raw: str | None) -> str:
    value = (raw or "bash").strip().lower()
    if value not in ALLOWED_INTERPRETERS:
        raise ScriptRegistrationError(
            f"unsupported interpreter: {raw}. Use one of {sorted(ALLOWED_INTERPRETERS)}"
        )
    return value


def _normalize_args(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScriptRegistrationError("default_args must be a list")
    return [str(item) for item in raw]


def _normalize_timeout(raw: Any) -> int:
    if raw is None:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ScriptRegistrationError("timeout_seconds must be a number") from exc
    if value <= 0:
        raise ScriptRegistrationError("timeout_seconds must be positive")
    # An unbounded script would hold its schedule's task "active" forever,
    # which is the stall the scheduler grace period exists to avoid.
    return min(value, _MAX_TIMEOUT_SECONDS)


class ScriptStore:
    """CRUD over the registered-script table."""

    def register(
        self,
        *,
        name: str,
        path: str,
        description: str | None = None,
        interpreter: str | None = None,
        default_args: Any = None,
        timeout_seconds: Any = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ScriptRegistrationError("script name is required")
        clean_path = _normalize_path(path)
        script_id = _new_id()
        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM agent_scripts WHERE path = ?", (clean_path,)
            ).fetchone()
            if existing is not None:
                raise ScriptRegistrationError(f"script already registered: {clean_path}")
            conn.execute(
                """
                INSERT INTO agent_scripts (
                    id, name, description, path, interpreter,
                    default_args_json, timeout_seconds, created_by
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    script_id,
                    clean_name,
                    (description or "").strip() or None,
                    clean_path,
                    _normalize_interpreter(interpreter),
                    json.dumps(_normalize_args(default_args)),
                    _normalize_timeout(timeout_seconds),
                    created_by,
                ),
            )
            conn.commit()
        return self.get(script_id) or {}

    def get(self, script_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_scripts WHERE id = ?", (script_id,)
            ).fetchone()
        return _row_to_script(row) if row else None

    def list_scripts(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                "SELECT * FROM agent_scripts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_script(row) for row in rows]

    def update(self, script_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        current = self.get(script_id)
        if current is None:
            return None
        fields: list[str] = []
        values: list[Any] = []
        if "name" in changes:
            clean = str(changes["name"] or "").strip()
            if not clean:
                raise ScriptRegistrationError("script name is required")
            fields.append("name = ?")
            values.append(clean)
        if "description" in changes:
            fields.append("description = ?")
            values.append((str(changes["description"] or "").strip()) or None)
        if "path" in changes:
            fields.append("path = ?")
            values.append(_normalize_path(changes["path"]))
        if "interpreter" in changes:
            fields.append("interpreter = ?")
            values.append(_normalize_interpreter(changes["interpreter"]))
        if "default_args" in changes:
            fields.append("default_args_json = ?")
            values.append(json.dumps(_normalize_args(changes["default_args"])))
        if "timeout_seconds" in changes:
            fields.append("timeout_seconds = ?")
            values.append(_normalize_timeout(changes["timeout_seconds"]))
        if not fields:
            return current
        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(script_id)
        with get_db_connection() as conn:
            conn.execute(
                f"UPDATE agent_scripts SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )
            conn.commit()
        return self.get(script_id)

    def delete(self, script_id: str) -> bool:
        with get_db_connection() as conn:
            cursor = conn.execute("DELETE FROM agent_scripts WHERE id = ?", (script_id,))
            conn.commit()
        return cursor.rowcount > 0


_script_store: ScriptStore | None = None


def get_script_store() -> ScriptStore:
    global _script_store
    if _script_store is None:
        _script_store = ScriptStore()
    return _script_store


__all__ = [
    "ALLOWED_INTERPRETERS",
    "ScriptRegistrationError",
    "ScriptStore",
    "get_script_store",
]
