"""Capability registry persistence mixin for AgentStore.

The capability registry is the catalog of tools the cockpit can route
through — Code Bridge built-ins, MCP servers, skills, native provider
tools, and CLI providers. It lives in ``agent_capabilities`` with its
own upsert / read verbs that live separately from run/task state.

This mixin owns those three methods. ``AgentStore`` composes it in so
callers keep using ``get_agent_store().upsert_capability(...)`` etc.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from core.database import get_db_connection

from ._row_converters import _row_to_capability


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class CapabilityStoreMixin:
    """Mixin contributing capability-registry methods to AgentStore."""

    def upsert_capability(
        self,
        *,
        capability_type: str,
        name: str,
        provider_id: str | None = None,
        status: str = "available",
        scope: str = "global",
        source: str = "codebridge",
        description: str | None = None,
        permission_level: str = "approval",
        desktop_only: bool = False,
        local_only: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        capability_id = (
            "cap_"
            + uuid.uuid5(
                uuid.NAMESPACE_URL,
                "|".join([capability_type, name, provider_id or "", source]),
            ).hex
        )
        with get_db_connection() as conn:
            existing = conn.execute(
                """
                SELECT id FROM agent_capabilities
                WHERE type = ?
                  AND name = ?
                  AND IFNULL(provider_id, '') = ?
                  AND source = ?
                """,
                (capability_type, name, provider_id or "", source),
            ).fetchone()
            if existing:
                capability_id = str(existing[0])
                conn.execute(
                    """
                    UPDATE agent_capabilities
                    SET status = ?,
                        scope = ?,
                        description = ?,
                        permission_level = ?,
                        desktop_only = ?,
                        local_only = ?,
                        metadata_json = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        status,
                        scope,
                        description,
                        permission_level,
                        1 if desktop_only else 0,
                        1 if local_only else 0,
                        _json_dumps(metadata or {}),
                        capability_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO agent_capabilities (
                        id, type, name, provider_id, status, scope, source,
                        description, permission_level, desktop_only, local_only,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        capability_id,
                        capability_type,
                        name,
                        provider_id,
                        status,
                        scope,
                        source,
                        description,
                        permission_level,
                        1 if desktop_only else 0,
                        1 if local_only else 0,
                        _json_dumps(metadata or {}),
                    ),
                )
            conn.commit()
        return self.get_capability(capability_id) or {}

    def get_capability(self, capability_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_capabilities WHERE id = ?",
                (capability_id,),
            ).fetchone()
        return _row_to_capability(row) if row else None

    def list_capabilities(
        self,
        *,
        capability_type: str | None = None,
        provider_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if capability_type:
            clauses.append("type = ?")
            values.append(capability_type)
        if provider_id:
            clauses.append("provider_id = ?")
            values.append(provider_id)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 500)))
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM agent_capabilities
                {where_sql}
                ORDER BY type ASC, provider_id ASC, name ASC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_capability(row) for row in rows]
