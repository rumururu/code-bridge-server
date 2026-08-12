"""Messages an unattended agent leaves for whoever comes back.

A scheduled run finishes while nobody is watching, so "tell me if the disk is
nearly full" cannot be a print statement — it has to survive until the phone
next asks. Delivery is deliberately a server capability rather than a script:
a script would make every agent that needs to say something depend on a
notifier the user first has to write, vet and register, and the message would
then be as reliable as that script.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.database import get_db_connection
from core.timestamps import to_utc_iso

ALLOWED_LEVELS = {"info", "success", "warning", "error"}
_MAX_TITLE = 120
_MAX_BODY = 4000


def _row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "task_id": row["task_id"],
        "agent_id": row["agent_id"],
        "title": row["title"],
        "body": row["body"],
        "level": row["level"],
        "read_at": to_utc_iso(row["read_at"]),
        "created_at": to_utc_iso(row["created_at"]),
    }


def normalize_level(raw: Any) -> str:
    value = str(raw or "info").strip().lower()
    return value if value in ALLOWED_LEVELS else "info"


class NotificationStore:
    def create(
        self,
        *,
        title: str,
        body: str | None = None,
        level: str = "info",
        run_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        clean_title = str(title or "").strip()[:_MAX_TITLE]
        if not clean_title:
            raise ValueError("notification title is required")
        notification_id = f"notif_{uuid.uuid4().hex}"
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_notifications (
                    id, run_id, task_id, agent_id, title, body, level
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notification_id,
                    run_id,
                    task_id,
                    agent_id,
                    clean_title,
                    (str(body).strip()[:_MAX_BODY] if body else None),
                    normalize_level(level),
                ),
            )
            conn.commit()
        return self.get(notification_id) or {}

    def get(self, notification_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_notifications WHERE id = ?", (notification_id,)
            ).fetchone()
        return _row(row) if row else None

    def list_notifications(
        self,
        *,
        unread_only: bool = False,
        agent_id: str | None = None,
        level: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if unread_only:
            clauses.append("read_at IS NULL")
        if agent_id:
            clauses.append("agent_id = ?")
            values.append(agent_id)
        if level:
            # Lets a caller ask "when did this agent last hear about a
            # failure" (level="error") separately from "...about needing a
            # human" (level="warning") without pulling every notification
            # and filtering in Python — see the notification throttles in
            # agent.task_orchestrator.
            clauses.append("level = ?")
            values.append(normalize_level(level))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 200)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"SELECT * FROM agent_notifications {where} "
                "ORDER BY created_at DESC LIMIT ?",
                tuple(values),
            ).fetchall()
        return [_row(row) for row in rows]

    def mark_read(self, notification_id: str) -> dict[str, Any] | None:
        with get_db_connection() as conn:
            # Only the first read stamps a time; re-reading must not move it,
            # or "when did I see this" stops meaning anything.
            conn.execute(
                "UPDATE agent_notifications SET read_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND read_at IS NULL",
                (notification_id,),
            )
            conn.commit()
        return self.get(notification_id)

    def mark_all_read(self) -> int:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "UPDATE agent_notifications SET read_at = CURRENT_TIMESTAMP "
                "WHERE read_at IS NULL"
            )
            conn.commit()
        return cursor.rowcount

    def unread_count(self) -> int:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM agent_notifications WHERE read_at IS NULL"
            ).fetchone()
        return int(row[0]) if row else 0


_notification_store: NotificationStore | None = None


def get_notification_store() -> NotificationStore:
    global _notification_store
    if _notification_store is None:
        _notification_store = NotificationStore()
    return _notification_store


__all__ = [
    "ALLOWED_LEVELS",
    "NotificationStore",
    "get_notification_store",
    "normalize_level",
]
