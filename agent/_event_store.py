"""Run-event persistence mixin for AgentStore.

``agent_events`` is the append-only timeline that powers the cockpit's
``/ws/agent/runs/{run_id}`` stream. Three methods own its lifecycle:
append + read by id + list-since-sequence for the WS polling loop.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.database import get_db_connection

from ._row_converters import _row_to_event


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class EventStoreMixin:
    """Mixin contributing event methods to AgentStore.

    Depends on ``self.get_run`` and ``self.get_event`` being provided by
    the surrounding store class.
    """

    def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        provider_id: str | None = None,
        provider_event: dict[str, Any] | None = None,
        app_event: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.get_run(run_id) is None:  # type: ignore[attr-defined]
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
                    _json_dumps(provider_event)
                    if provider_event is not None
                    else None,
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
