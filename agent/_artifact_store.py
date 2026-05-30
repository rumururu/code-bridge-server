"""Artifact persistence mixin for AgentStore.

``agent_artifacts`` stores tool outputs (terminal logs, source diffs,
preview screenshots, APK / IPA manifest snapshots, visual regression
diffs, build-output sizing, ...) durably linked to the run that produced
them. Three methods: add + read by id + list per run.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.database import get_db_connection

from ._row_converters import _row_to_artifact


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class ArtifactStoreMixin:
    """Mixin contributing artifact methods to AgentStore.

    Depends on ``self.get_run`` and ``self.get_artifact``.
    """

    def add_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        path: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.get_run(run_id) is None:  # type: ignore[attr-defined]
            return None
        artifact_id = _new_id("art")
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_artifacts (id, run_id, kind, path, mime_type, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    run_id,
                    kind,
                    path,
                    mime_type,
                    _json_dumps(metadata or {}),
                ),
            )
            conn.execute(
                "UPDATE agent_runs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,),
            )
            conn.commit()
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()
        return _row_to_artifact(row) if row else None

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_artifacts
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [_row_to_artifact(row) for row in rows]
