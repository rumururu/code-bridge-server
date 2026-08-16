"""Persistent policy rule storage and evaluation."""

import json
import uuid
from typing import Any

from core.database import get_db_connection, init_db

from .policy_engine import decide_policy
from core.timestamps import to_utc_iso

ALLOWED_RULE_EFFECTS = {"allow", "confirm_once", "confirm_each", "desktop_only", "forbidden"}

#: The ``project_name`` a project-less run carries — ``GLOBAL_TASK_PROJECT_NAME``
#: in ``agent/task_orchestrator.py``. Duplicated as a literal rather than
#: imported because the policy layer must not depend on the agent runtime.
#:
#: It is not a project. Every project-less agent on the server shares it, so a
#: rule at ``project:__global__`` would be a near-global grant wearing a
#: project's name. The write side (``standing_rule_scope_for_request``) has
#: always refused to *create* one; :func:`_scope_candidates` used to still
#: *match* one, so a rule written there by hand (the dashboard's free-text
#: scope box, or the rules API) applied to every project-less run on the box.
#: Both sides now go through :func:`scope_project_name`, which is the only
#: way the two can't drift apart again.
GLOBAL_PROJECT_SENTINEL = "__global__"
_EFFECT_RISK = {
    "allow": "low",
    "confirm_once": "medium",
    "confirm_each": "high",
    "desktop_only": "high",
    "forbidden": "critical",
}


def _new_id() -> str:
    return f"pol_{uuid.uuid4().hex}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _row_to_rule(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "scope": row["scope"],
        "operation": row["operation"],
        "effect": row["effect"],
        "constraints": _json_loads(row["constraints_json"], {}),
        "created_by": row["created_by"],
        "created_at": to_utc_iso(row["created_at"]),
        "expires_at": to_utc_iso(row["expires_at"]),
    }


class PolicyRuleStore:
    """SQLite-backed persistent policy rule helper."""

    def __init__(self) -> None:
        init_db()

    def create_rule(
        self,
        *,
        scope: str,
        operation: str,
        effect: str,
        constraints: dict[str, Any] | None = None,
        created_by: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        normalized_effect = effect.strip().lower()
        if normalized_effect not in ALLOWED_RULE_EFFECTS:
            raise ValueError(f"Unsupported policy effect: {effect}")
        rule_id = _new_id()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO policy_rules (
                    id, scope, operation, effect, constraints_json,
                    created_by, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule_id,
                    scope.strip() or "global",
                    operation.strip(),
                    normalized_effect,
                    _json_dumps(constraints or {}),
                    created_by,
                    expires_at,
                ),
            )
            conn.commit()
        return self.get_rule(rule_id) or {}

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM policy_rules WHERE id = ?",
                (rule_id,),
            ).fetchone()
        return _row_to_rule(row) if row else None

    def list_rules(
        self,
        *,
        scope: str | None = None,
        operation: str | None = None,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if scope:
            clauses.append("scope = ?")
            values.append(scope)
        if operation:
            clauses.append("operation = ?")
            values.append(operation)
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM policy_rules
                {where_sql}
                ORDER BY created_at DESC, rowid DESC
                """,
                values,
            ).fetchall()
        return [_row_to_rule(row) for row in rows]

    def delete_rule(self, rule_id: str) -> bool:
        with get_db_connection() as conn:
            cursor = conn.execute("DELETE FROM policy_rules WHERE id = ?", (rule_id,))
            conn.commit()
            return cursor.rowcount > 0

    def find_effective_rule(
        self,
        *,
        operation: str,
        scope_candidates: list[str],
    ) -> dict[str, Any] | None:
        scopes = [scope for scope in scope_candidates if scope]
        if not scopes:
            scopes = ["global"]
        placeholders = ",".join("?" for _ in scopes)
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM policy_rules
                WHERE operation = ?
                  AND scope IN ({placeholders})
                  AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY
                  CASE scope
                    {' '.join(f"WHEN ? THEN {index}" for index, _ in enumerate(scopes))}
                    ELSE {len(scopes)}
                  END,
                  rowid DESC
                LIMIT 1
                """,
                [operation, *scopes, *scopes],
            ).fetchall()
        return _row_to_rule(rows[0]) if rows else None


def decide_policy_with_rules(
    operation: str,
    *,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply built-in policy and the safest matching persistent rule."""
    details = details or {}
    base = decide_policy(operation, details=details)
    if base["forbidden"] or base["desktop_only"]:
        return base

    rule = get_policy_rule_store().find_effective_rule(
        operation=base["operation"],
        scope_candidates=_scope_candidates(run_id=run_id, details=details),
    )
    if rule is None:
        return base

    effect = rule["effect"]
    return {
        **base,
        "effect": effect,
        "risk_level": _EFFECT_RISK.get(effect, base["risk_level"]),
        "reason": f"Persistent policy rule {rule['id']} matched",
        "approval_required": effect in {"confirm_once", "confirm_each", "desktop_only"},
        "desktop_only": effect == "desktop_only",
        "forbidden": effect == "forbidden",
        "rule": rule,
    }


def _clean(value: Any) -> str | None:
    """``value`` as a non-empty stripped string, else ``None``."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def scope_agent_id(details: dict[str, Any]) -> str | None:
    """The agent a standing rule for ``details`` may be anchored to.

    ``None`` when the operation did not come from a nameable agent, which is
    the honest answer for interactive chat: those runs are all filed under one
    pseudo-agent, so an ``agent:`` rule there would grant the operation to
    every chat on the server. ``chat/chat_stream_service.py`` only puts
    ``agent_id`` in the details when it resolves to a real, non-pseudo agent —
    this reads what it wrote.
    """
    return _clean(details.get("agent_id"))


def scope_workspace_id(details: dict[str, Any]) -> str | None:
    """The workspace a standing rule for ``details`` may be anchored to."""
    return _clean(details.get("workspace_id"))


def scope_project_name(details: dict[str, Any]) -> str | None:
    """The project a standing rule for ``details`` may be anchored to.

    ``None`` for :data:`GLOBAL_PROJECT_SENTINEL`, which names no project.
    """
    project_name = _clean(details.get("project_name"))
    if project_name == GLOBAL_PROJECT_SENTINEL:
        return None
    return project_name


def _scope_candidates(
    *,
    run_id: str | None,
    details: dict[str, Any],
) -> list[str]:
    """Scopes a rule may be filed under, narrowest first.

    ``find_effective_rule`` ranks by position, so the first entry that has a
    rule wins. The order is the exact inverse of the one
    ``approvals.approval_service.standing_rule_scope_for_request`` writes at —
    a rule written at the scope that function picks is found here, which is
    the only property that makes "always allow" mean anything.

    ``agent:`` sits between the run and the workspace: it is one named actor
    rather than everything that happens in a place, and it is what survives a
    schedule — a new fire is a new run, so a ``run:`` rule is dead on arrival
    for the unattended case it was granted for.
    """
    candidates: list[str] = []
    if run_id:
        candidates.append(f"run:{run_id}")
    agent_id = scope_agent_id(details)
    if agent_id:
        candidates.append(f"agent:{agent_id}")
    workspace_id = scope_workspace_id(details)
    if workspace_id:
        candidates.append(f"workspace:{workspace_id}")
    project_name = scope_project_name(details)
    if project_name:
        candidates.append(f"project:{project_name}")
    candidates.append("global")
    return candidates


_policy_rule_store: PolicyRuleStore | None = None


def get_policy_rule_store() -> PolicyRuleStore:
    """Return the process-global policy rule store."""
    global _policy_rule_store
    if _policy_rule_store is None:
        _policy_rule_store = PolicyRuleStore()
    return _policy_rule_store
