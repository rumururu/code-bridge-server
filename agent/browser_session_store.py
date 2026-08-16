"""SQLite store for browser handoff sessions."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.database import get_db_connection, init_db
from core.runtime_paths import SERVER_DIR, runtime_dir

DEFAULT_BROWSER_SESSION_TTL_MINUTES = 30
BROWSER_SESSION_ROOT = runtime_dir(
    "browser_sessions",
    SERVER_DIR / "core" / "browser_sessions",
)

_browser_session_store: "BrowserSessionStore | None" = None


def _new_id() -> str:
    return f"bs_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _row_to_browser_session(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "task_id": row["task_id"],
        "step_id": row["step_id"],
        "workflow_step_id": row["workflow_step_id"],
        "status": row["status"],
        "context_dir": row["context_dir"],
        "storage_state_path": row["storage_state_path"],
        "current_url": row["current_url"],
        "title": row["title"],
        "handoff_reason": row["handoff_reason"],
        "metadata": _json_loads(row["metadata_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "expires_at": row["expires_at"],
        "closed_at": row["closed_at"],
    }


class BrowserSessionStore:
    """Persistence helper for browser session handoff state."""

    def __init__(self) -> None:
        init_db()

    def create(
        self,
        *,
        run_id: str,
        task_id: str | None = None,
        step_id: str | None = None,
        workflow_step_id: str | None = None,
        status: str = "created",
        current_url: str | None = None,
        title: str | None = None,
        handoff_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_minutes: int = DEFAULT_BROWSER_SESSION_TTL_MINUTES,
    ) -> dict[str, Any]:
        session_id = _new_id()
        context_dir = str((BROWSER_SESSION_ROOT / session_id).resolve())
        expires_at = _utcnow() + timedelta(minutes=max(1, int(ttl_minutes)))
        Path(context_dir).mkdir(parents=True, exist_ok=True)
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO browser_sessions (
                    id, run_id, task_id, step_id, workflow_step_id, status,
                    context_dir, current_url, title, handoff_reason,
                    metadata_json, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    run_id,
                    task_id,
                    step_id,
                    workflow_step_id,
                    status,
                    context_dir,
                    current_url,
                    title,
                    handoff_reason,
                    _json_dumps(metadata or {}),
                    _iso(expires_at),
                ),
            )
            conn.commit()
        session = self.get(session_id)
        assert session is not None
        return session

    def get(self, session_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM browser_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return _row_to_browser_session(row)

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM browser_sessions
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [session for row in rows if (session := _row_to_browser_session(row))]

    def find_active_for_step(
        self,
        *,
        run_id: str,
        step_id: str,
    ) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                """
                SELECT * FROM browser_sessions
                WHERE run_id = ?
                  AND step_id = ?
                  AND status IN ('created', 'waiting_for_user', 'resumed')
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (run_id, step_id),
            ).fetchone()
        return _row_to_browser_session(row)

    def latest_resumable_for_run(self, run_id: str) -> dict[str, Any] | None:
        # `rowid DESC` is the tiebreak, not decoration: SQLite's
        # CURRENT_TIMESTAMP has one-second resolution, and two browser steps of
        # one run land inside the same second routinely. Without it "newest"
        # is undefined exactly where it is asked most often.
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                """
                SELECT * FROM browser_sessions
                WHERE run_id = ?
                  AND status IN ('created', 'waiting_for_user', 'resumed')
                  AND storage_state_path IS NOT NULL
                ORDER BY updated_at DESC, created_at DESC, rowid DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return _row_to_browser_session(row)

    def latest_with_storage_state_for_task(
        self,
        task_id: str,
        *,
        exclude_run_id: str | None = None,
        exclude_session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """The newest session of this *task* that left a storage state behind.

        Scoped to the task, not the run, and that is the whole point.
        `latest_resumable_for_run` only ever looks inside the run being
        executed, so a scheduled agent — a new run every night — started signed
        out however many times a person had logged in. The task id is what
        survives between runs, so a login captured on Monday is the input to
        Tuesday's run.

        Returns the whole session rather than just the path so the caller can
        name which session it is continuing from. A path on its own would leave
        `browser_input_storage_state_path` and `previous_browser_session_id`
        free to disagree about where the login came from.

        Closed and expired sessions are included on purpose: a finished session
        is exactly where a completed login lives.

        `exclude_session_id` is how a session is kept from reading back its own
        state. It is deliberately narrower than `exclude_run_id`: excluding the
        whole current run would also skip an earlier *step* of that run, whose
        state is newer than any previous run's and is the one to continue from.
        Ordering does the rest — newest wins, so an earlier step in this run
        beats last night, and last night wins when this run has nothing yet.
        """
        clauses = [
            "task_id = ?",
            "storage_state_path IS NOT NULL",
            "storage_state_path != ''",
        ]
        values: list[Any] = [task_id]
        if exclude_run_id:
            clauses.append("run_id != ?")
            values.append(exclude_run_id)
        if exclude_session_id:
            clauses.append("id != ?")
            values.append(exclude_session_id)
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                f"""
                SELECT * FROM browser_sessions
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, created_at DESC, rowid DESC
                LIMIT 1
                """,
                tuple(values),
            ).fetchone()
        return _row_to_browser_session(row)

    def latest_storage_state_for_task(
        self,
        task_id: str,
        *,
        exclude_run_id: str | None = None,
        exclude_session_id: str | None = None,
    ) -> str | None:
        """Just the path from :meth:`latest_with_storage_state_for_task`.

        One SQL statement behind both so the adapter's own safety net and the
        orchestrator's handoff can never select different sessions.
        """
        session = self.latest_with_storage_state_for_task(
            task_id,
            exclude_run_id=exclude_run_id,
            exclude_session_id=exclude_session_id,
        )
        if session is None:
            return None
        value = session.get("storage_state_path")
        return str(value) if value else None

    def update(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        allowed = {
            "status": "status",
            "storage_state_path": "storage_state_path",
            "current_url": "current_url",
            "title": "title",
            "handoff_reason": "handoff_reason",
            "metadata": "metadata_json",
            "expires_at": "expires_at",
            "closed_at": "closed_at",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in updates.items():
            column = allowed.get(key)
            if column is None:
                raise ValueError(f"Unsupported browser session field: {key}")
            assignments.append(f"{column} = ?")
            if key == "metadata":
                values.append(_json_dumps(value or {}))
            elif key in {"expires_at", "closed_at"} and isinstance(value, datetime):
                values.append(_iso(value))
            else:
                values.append(value)
        if not assignments:
            return self.get(session_id)
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(session_id)
        with get_db_connection() as conn:
            conn.execute(
                f"""
                UPDATE browser_sessions
                SET {', '.join(assignments)}
                WHERE id = ?
                """,
                tuple(values),
            )
            conn.commit()
        return self.get(session_id)

    def mark_waiting(
        self,
        session_id: str,
        *,
        reason: str | None = None,
        current_url: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any] | None:
        return self.update(
            session_id,
            {
                "status": "waiting_for_user",
                "handoff_reason": reason,
                "current_url": current_url,
                "title": title,
            },
        )

    def mark_resumed(self, session_id: str) -> dict[str, Any] | None:
        return self.update(session_id, {"status": "resumed"})

    def close(self, session_id: str) -> dict[str, Any] | None:
        return self.update(
            session_id,
            {
                "status": "closed",
                "closed_at": _utcnow(),
            },
        )

    def expire_due(self, *, now: datetime | None = None) -> int:
        current = now or _utcnow()
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE browser_sessions
                SET status = 'expired',
                    closed_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status IN ('created', 'waiting_for_user', 'resumed')
                  AND expires_at IS NOT NULL
                  AND expires_at <= ?
                """,
                (_iso(current), _iso(current)),
            )
            conn.commit()
            return int(cursor.rowcount or 0)


def get_browser_session_store() -> BrowserSessionStore:
    global _browser_session_store
    if _browser_session_store is None:
        _browser_session_store = BrowserSessionStore()
    return _browser_session_store
