"""DAO for task_schedules.

Task schedules are durable plans that fire a stored task through the same
prepare_task_orchestration / execute_task_orchestration pipeline as a manual
``POST /api/agent/tasks/{task_id}/start`` call. Provider, model, cwd, prompt,
and requested capabilities are captured at schedule-creation time so the
scheduler can replay the exact intent without prompting the user again.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.database import get_db_connection
from core.timestamps import to_utc_iso

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _validate_expression(expression: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a schedule expression.

    Two kinds are supported:

    ``{"kind": "interval", "seconds": N}``      — fires every N seconds, N>=60
    ``{"kind": "daily_at", "time": "HH:MM"}``   — fires once a day at HH:MM local
    """
    if not isinstance(expression, dict):
        raise ValueError("expression must be an object")
    kind = expression.get("kind")
    if kind == "interval":
        seconds = expression.get("seconds")
        if not isinstance(seconds, int) or seconds < 60:
            raise ValueError("interval.seconds must be an integer >= 60")
        return {"kind": "interval", "seconds": seconds}
    if kind == "daily_at":
        raw = expression.get("time")
        if not isinstance(raw, str):
            raise ValueError("daily_at.time must be 'HH:MM'")
        try:
            hh, mm = raw.split(":", 1)
            hour = int(hh)
            minute = int(mm)
        except (TypeError, ValueError) as exc:
            raise ValueError("daily_at.time must be 'HH:MM'") from exc
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError("daily_at.time out of range")
        return {"kind": "daily_at", "time": f"{hour:02d}:{minute:02d}"}
    raise ValueError(f"unsupported schedule kind: {kind!r}")


def compute_next_fire(
    expression: dict[str, Any],
    *,
    after: datetime | None = None,
) -> datetime:
    """Compute the next firing time given a (validated) expression."""
    base = after or _utcnow()
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    kind = expression["kind"]
    if kind == "interval":
        return base + timedelta(seconds=expression["seconds"])
    if kind == "daily_at":
        # Local time semantics: HH:MM in the server's local timezone.
        local_now = base.astimezone()  # convert to server local tz
        hour, minute = (int(part) for part in expression["time"].split(":"))
        candidate = local_now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )
        if candidate <= local_now:
            candidate = candidate + timedelta(days=1)
        return candidate.astimezone(timezone.utc)
    raise ValueError(f"unsupported schedule kind: {kind!r}")


def compute_next_fire_at(
    agent_id: str,
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Return the nearest enabled schedule fire time for an agent."""
    base = now or _utcnow()
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    candidates: list[datetime] = []
    with get_db_connection(use_row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT s.expression_json, s.next_run_at
            FROM task_schedules s
            JOIN agent_tasks t ON t.id = s.task_id
            WHERE s.enabled = 1
              AND t.assigned_agent_id = ?
            """,
            (agent_id,),
        ).fetchall()

    for row in rows:
        next_at = _evaluate_next_fire(row, after=base)
        if next_at is not None:
            candidates.append(next_at)
    return min(candidates) if candidates else None


def _evaluate_next_fire(row: sqlite3.Row, *, after: datetime) -> datetime | None:
    stored = _parse_dt(row["next_run_at"])
    if stored is not None and stored > after:
        return stored.astimezone(timezone.utc)
    try:
        expression = json.loads(row["expression_json"])
        return compute_next_fire(expression, after=after)
    except (TypeError, ValueError, json.JSONDecodeError, KeyError):
        logger.warning("failed to evaluate schedule expression for next fire")
        return None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "name": row["name"],
        "expression": json.loads(row["expression_json"]),
        "provider_id": row["provider_id"],
        "model": row["model"],
        "cwd": row["cwd"],
        "prompt": row["prompt"],
        "capabilities": json.loads(row["capabilities_json"]),
        "enabled": bool(row["enabled"]),
        "skip_if_active": bool(row["skip_if_active"]),
        "next_run_at": to_utc_iso(row["next_run_at"]),
        "last_run_at": to_utc_iso(row["last_run_at"]),
        "last_run_id": row["last_run_id"],
        "last_status": row["last_status"],
        "last_error": row["last_error"],
        "fire_count": int(row["fire_count"] or 0),
        "skip_count": int(row["skip_count"] or 0),
        "created_at": to_utc_iso(row["created_at"]),
        "updated_at": to_utc_iso(row["updated_at"]),
    }


class TaskScheduleStore:
    """Durable storage for scheduled task firings."""

    def create(
        self,
        *,
        task_id: str,
        expression: dict[str, Any],
        name: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
        prompt: str | None = None,
        capabilities: list[str] | None = None,
        enabled: bool = True,
        skip_if_active: bool = True,
    ) -> dict[str, Any]:
        validated = _validate_expression(expression)
        schedule_id = uuid.uuid4().hex
        next_run = compute_next_fire(validated)
        with get_db_connection(use_row_factory=True) as conn:
            conn.execute(
                """
                INSERT INTO task_schedules (
                    id, task_id, name, expression_json,
                    provider_id, model, cwd, prompt, capabilities_json,
                    enabled, skip_if_active, next_run_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id,
                    task_id,
                    name,
                    json.dumps(validated),
                    provider_id,
                    model,
                    cwd,
                    prompt,
                    json.dumps(list(capabilities or [])),
                    1 if enabled else 0,
                    1 if skip_if_active else 0,
                    _iso(next_run),
                ),
            )
            conn.commit()
        fetched = self.get(schedule_id)
        assert fetched is not None
        return fetched

    def get(self, schedule_id: str) -> Optional[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM task_schedules WHERE id = ?",
                (schedule_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_for_task(self, task_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM task_schedules
                WHERE task_id = ?
                ORDER BY created_at DESC
                """,
                (task_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_all(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM task_schedules"
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY next_run_at IS NULL, next_run_at ASC"
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(sql).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_due(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        cutoff = _iso(now or _utcnow())
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM task_schedules
                WHERE enabled = 1
                  AND next_run_at IS NOT NULL
                  AND next_run_at <= ?
                ORDER BY next_run_at ASC
                """,
                (cutoff,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update(
        self,
        schedule_id: str,
        updates: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Patch a schedule. Recomputes ``next_run_at`` if expression changes."""
        if not updates:
            return self.get(schedule_id)

        allowed = {
            "name",
            "provider_id",
            "model",
            "cwd",
            "prompt",
            "capabilities",
            "enabled",
            "skip_if_active",
            "expression",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unknown update fields: {sorted(unknown)}")

        columns: list[str] = []
        values: list[Any] = []

        if "name" in updates:
            columns.append("name = ?")
            values.append(updates["name"])
        if "provider_id" in updates:
            columns.append("provider_id = ?")
            values.append(updates["provider_id"])
        if "model" in updates:
            columns.append("model = ?")
            values.append(updates["model"])
        if "cwd" in updates:
            columns.append("cwd = ?")
            values.append(updates["cwd"])
        if "prompt" in updates:
            columns.append("prompt = ?")
            values.append(updates["prompt"])
        if "capabilities" in updates:
            columns.append("capabilities_json = ?")
            values.append(json.dumps(list(updates["capabilities"] or [])))
        if "enabled" in updates:
            columns.append("enabled = ?")
            values.append(1 if updates["enabled"] else 0)
        if "skip_if_active" in updates:
            columns.append("skip_if_active = ?")
            values.append(1 if updates["skip_if_active"] else 0)
        if "expression" in updates:
            validated = _validate_expression(updates["expression"])
            columns.append("expression_json = ?")
            values.append(json.dumps(validated))
            columns.append("next_run_at = ?")
            values.append(_iso(compute_next_fire(validated)))

        columns.append("updated_at = CURRENT_TIMESTAMP")
        values.append(schedule_id)

        with get_db_connection(use_row_factory=True) as conn:
            conn.execute(
                f"UPDATE task_schedules SET {', '.join(columns)} WHERE id = ?",
                values,
            )
            conn.commit()
        return self.get(schedule_id)

    def delete(self, schedule_id: str) -> bool:
        with get_db_connection() as conn:
            cur = conn.execute(
                "DELETE FROM task_schedules WHERE id = ?",
                (schedule_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    def record_fire(
        self,
        schedule_id: str,
        *,
        run_id: str | None,
        status: str,
        error: str | None = None,
        next_run_at: datetime | None = None,
    ) -> None:
        """Record the outcome of a firing attempt and advance ``next_run_at``."""
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT expression_json FROM task_schedules WHERE id = ?",
                (schedule_id,),
            ).fetchone()
            if row is None:
                return
            expression = json.loads(row["expression_json"])
            advanced = next_run_at or compute_next_fire(expression)
            increment_fire = 1 if status == "fired" else 0
            increment_skip = 1 if status == "skipped" else 0
            conn.execute(
                """
                UPDATE task_schedules
                SET last_run_at = ?,
                    last_run_id = ?,
                    last_status = ?,
                    last_error = ?,
                    fire_count = fire_count + ?,
                    skip_count = skip_count + ?,
                    next_run_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    _iso(_utcnow()),
                    run_id,
                    status,
                    error,
                    increment_fire,
                    increment_skip,
                    _iso(advanced),
                    schedule_id,
                ),
            )
            conn.commit()


_store: TaskScheduleStore | None = None


def get_schedule_store() -> TaskScheduleStore:
    global _store
    if _store is None:
        _store = TaskScheduleStore()
    return _store


# Re-export for tests that want monotonic timestamps without importing time.
def monotonic() -> float:
    return time.monotonic()
