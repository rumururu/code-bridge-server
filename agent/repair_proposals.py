"""Repair proposals (AGENT_SELF_REPAIR_SPEC §2, §4).

When a run fails, the server may propose a fix — a revised flow it could
apply on a tap, or a description of what a person has to do — and keep it
until someone applies it, rejects it, or seven days pass. The server never
applies one itself (ADR-003): ``accept`` goes through ``update_agent`` with
the proposal's ``base_flow_revision`` as the precondition, exactly the save a
person makes from the canvas.

Bounded on purpose: one open proposal per agent (a repeating failure joins
it as ``seen_runs``), a daily cap, no proposal for a run the server itself
interrupted or for a failure that is an unanswered person.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from core.database import get_db_connection

logger = logging.getLogger(__name__)

STATUS_PROPOSED = "proposed"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_SUPERSEDED = "superseded"
STATUS_EXPIRED = "expired"

KIND_WORKFLOW = "workflow"
KIND_GENERATED_SCRIPT = "generated_script"
KIND_SCRIPT_CHANGE = "script_change"
KIND_ENVIRONMENT = "environment"
KIND_UNKNOWN = "unknown"
KINDS = (KIND_WORKFLOW, KIND_GENERATED_SCRIPT, KIND_SCRIPT_CHANGE, KIND_ENVIRONMENT, KIND_UNKNOWN)
APPLICABLE_KINDS = frozenset({KIND_WORKFLOW, KIND_GENERATED_SCRIPT})

#: Diagnosis classes that are not a configuration problem.
NOT_A_CONFIG_PROBLEM = frozenset({"interrupted", "approval_denied", "approval_expired", "park_abandoned"})

EXPIRY_DAYS = 7
NOTIFICATION_REASON = "repair_proposal"


def proposals_per_day() -> int:
    raw = os.environ.get("CODEBRIDGE_REPAIR_PROPOSALS_PER_DAY", "3")
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def enabled() -> bool:
    return os.environ.get("CODEBRIDGE_REPAIR_PROPOSALS", "1") not in ("0", "false", "no")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "agent_id": row["agent_id"],
        "run_id": row["run_id"],
        "seen_runs": _loads(row["seen_runs_json"], []),
        "base_flow_revision": row["base_flow_revision"],
        "briefing": _loads(row["briefing_json"], {}),
        "diagnosis": _loads(row["diagnosis_json"], {}),
        "kind": row["kind"],
        "summary": row["summary"],
        "flow_json": _loads(row["flow_json"], None),
        "human_actions": _loads(row["human_actions_json"], []),
        "warnings": _loads(row["warnings_json"], []),
        "status": row["status"],
        "reject_reason": row["reject_reason"],
        "applied_flow_revision": row["applied_flow_revision"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
        "resolved_by": row["resolved_by"],
        "applicable": row["kind"] in APPLICABLE_KINDS and row["status"] == STATUS_PROPOSED and row["flow_json"] is not None,
    }


class RepairProposalStore:
    def create(
        self,
        *,
        agent_id: str,
        run_id: str | None,
        base_flow_revision: str | None,
        briefing: dict[str, Any],
        diagnosis: dict[str, Any],
        kind: str,
        summary: str,
        flow_json: list[dict[str, Any]] | None,
        human_actions: list[dict[str, Any]],
        warnings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        proposal_id = f"rp_{uuid.uuid4().hex}"
        with get_db_connection(use_row_factory=True) as conn:
            conn.execute(
                """
                INSERT INTO agent_repair_proposals (
                    id, agent_id, run_id, seen_runs_json, base_flow_revision, briefing_json,
                    diagnosis_json, kind, summary, flow_json, human_actions_json, warnings_json,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id, agent_id, run_id, json.dumps([run_id] if run_id else []),
                    base_flow_revision, json.dumps(briefing, ensure_ascii=False),
                    json.dumps(diagnosis, ensure_ascii=False), kind if kind in KINDS else KIND_UNKNOWN,
                    summary, json.dumps(flow_json, ensure_ascii=False) if flow_json is not None else None,
                    json.dumps(human_actions, ensure_ascii=False), json.dumps(warnings or [], ensure_ascii=False),
                    STATUS_PROPOSED, _now(),
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM agent_repair_proposals WHERE id = ?", (proposal_id,)).fetchone()
        return _row(row)

    def get(self, proposal_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute("SELECT * FROM agent_repair_proposals WHERE id = ?", (proposal_id,)).fetchone()
        return _row(row) if row else None

    def open_for_agent(self, agent_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_repair_proposals WHERE agent_id = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
                (agent_id, STATUS_PROPOSED),
            ).fetchone()
        return _row(row) if row else None

    def list_for_agent(self, agent_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                "SELECT * FROM agent_repair_proposals WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
                (agent_id, max(1, min(int(limit), 200))),
            ).fetchall()
        return [_row(r) for r in rows]

    def list_open(self) -> list[dict[str, Any]]:
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                "SELECT * FROM agent_repair_proposals WHERE status = ? ORDER BY created_at DESC", (STATUS_PROPOSED,)
            ).fetchall()
        return [_row(r) for r in rows]

    def created_today(self, agent_id: str) -> int:
        since = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
        with get_db_connection() as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM agent_repair_proposals WHERE agent_id = ? AND created_at >= ?",
                (agent_id, since),
            ).fetchone()
        return int(count)

    def add_seen_run(self, proposal_id: str, run_id: str) -> None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute("SELECT seen_runs_json FROM agent_repair_proposals WHERE id = ?", (proposal_id,)).fetchone()
            if not row:
                return
            seen = _loads(row["seen_runs_json"], [])
            if run_id not in seen:
                seen.append(run_id)
            conn.execute("UPDATE agent_repair_proposals SET seen_runs_json = ? WHERE id = ?", (json.dumps(seen), proposal_id))
            conn.commit()

    def resolve(
        self, proposal_id: str, *, status: str, resolved_by: str | None = None,
        reject_reason: str | None = None, applied_flow_revision: str | None = None,
    ) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            conn.execute(
                """
                UPDATE agent_repair_proposals
                SET status = ?, resolved_at = ?, resolved_by = ?, reject_reason = ?, applied_flow_revision = ?
                WHERE id = ? AND status = ?
                """,
                (status, _now(), resolved_by, reject_reason, applied_flow_revision, proposal_id, STATUS_PROPOSED),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM agent_repair_proposals WHERE id = ?", (proposal_id,)).fetchone()
        return _row(row) if row else None

    def supersede_stale(self, agent_id: str, current_flow_revision: str) -> list[dict[str, Any]]:
        """Open proposals whose base is no longer the stored flow are superseded."""
        out = []
        open_proposal = self.open_for_agent(agent_id)
        if open_proposal and open_proposal.get("base_flow_revision") not in (None, current_flow_revision):
            out.append(self.resolve(open_proposal["id"], status=STATUS_SUPERSEDED, resolved_by="flow_changed"))
        return [p for p in out if p]

    def expire_stale(self, *, days: int = EXPIRY_DAYS) -> list[dict[str, Any]]:
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
        with get_db_connection(use_row_factory=True) as conn:
            rows = conn.execute(
                "SELECT id FROM agent_repair_proposals WHERE status = ? AND created_at < ?", (STATUS_PROPOSED, cutoff)
            ).fetchall()
        return [p for p in (self.resolve(r["id"], status=STATUS_EXPIRED, resolved_by="sweep") for r in rows) if p]

    def last_rejected(self, agent_id: str) -> dict[str, Any] | None:
        with get_db_connection(use_row_factory=True) as conn:
            row = conn.execute(
                "SELECT * FROM agent_repair_proposals WHERE agent_id = ? AND status = ? ORDER BY resolved_at DESC LIMIT 1",
                (agent_id, STATUS_REJECTED),
            ).fetchone()
        return _row(row) if row else None


_store: RepairProposalStore | None = None


def get_repair_proposal_store() -> RepairProposalStore:
    global _store
    if _store is None:
        _store = RepairProposalStore()
    return _store


# --- kind classification (§4) ----------------------------------------------

def classify_kind(diagnosis: dict[str, Any] | None, briefing: dict[str, Any] | None, proposals: list[Any]) -> str:
    """Who can act on this proposal — the server on a tap, or a person."""
    suggested = (diagnosis or {}).get("suggested_change") or {}
    kind = str(suggested.get("kind") or "")
    if kind in (KIND_SCRIPT_CHANGE, KIND_ENVIRONMENT):
        return kind
    external = {s.get("script_id") for s in (briefing or {}).get("registered_scripts", []) if s.get("origin") == "registered"}
    failed_step = None
    for run in (briefing or {}).get("runs", []):
        if run.get("failed_step"):
            failed_step = run["failed_step"]
            break
    if proposals:
        return KIND_WORKFLOW
    if kind == KIND_GENERATED_SCRIPT:
        return KIND_GENERATED_SCRIPT
    if failed_step and failed_step.get("type") == "shell" and external:
        return KIND_SCRIPT_CHANGE
    return KIND_UNKNOWN


def human_actions_for(kind: str, diagnosis: dict[str, Any] | None, briefing: dict[str, Any] | None) -> list[dict[str, Any]]:
    suggested = (diagnosis or {}).get("suggested_change") or {}
    actions: list[dict[str, Any]] = []
    if kind in (KIND_SCRIPT_CHANGE, KIND_ENVIRONMENT, KIND_UNKNOWN) and suggested.get("what"):
        actions.append({"what": suggested.get("what"), "where": suggested.get("where") or "", "why": (diagnosis or {}).get("cause") or ""})
    if kind == KIND_SCRIPT_CHANGE:
        for s in (briefing or {}).get("registered_scripts", []):
            if s.get("origin") == "registered":
                actions.append({"what": f"edit script {s.get('name') or s.get('script_id')}", "where": s.get("path") or "", "why": "managed outside this server; the Configurator cannot change it"})
    return actions


# --- the trigger (§2.1–2.3) --------------------------------------------------

async def _suggest(agent: dict[str, Any], briefing: dict[str, Any], diagnosis: dict[str, Any]) -> tuple[list[Any], list[Any], str | None]:
    """One suggest call with the briefing attached; validated through the same gate as the canvas."""
    from code_bridge_core import graph_suggest
    from code_bridge_core.workflow_v2 import normalize_workflow
    from agent.run_diagnosis import _one_shot

    suggested = diagnosis.get("suggested_change") or {}
    intent = "Propose the smallest change to this workflow that addresses the most recent failure in the record above."
    if suggested.get("what"):
        intent += f" The diagnosis suggests: {suggested.get('what')}"
        if suggested.get("where"):
            intent += f" ({suggested.get('where')})"
    intent += " If the fix is not in the workflow (a script managed outside this server, the device, a permission), say so in the assessment and propose nothing."
    prompt = graph_suggest.build_suggest_prompt(
        agent_name=str(agent.get("name") or agent.get("id")),
        agent_system_prompt=str(agent.get("system_prompt") or ""),
        flow=normalize_workflow(agent.get("flow_json")),
        intent=intent,
        briefing=briefing,
    )
    raw = await _one_shot(prompt, timeout=120.0)
    parsed = graph_suggest.parse_suggest_response(raw)
    scripts = None
    try:
        from agent.script_store import get_script_store

        scripts = {s["id"]: s for s in get_script_store().list_scripts(limit=200)}
    except Exception:
        scripts = None
    proposals, dropped = graph_suggest.validate_proposals(parsed.proposals, scripts=scripts)
    return proposals, dropped, None


async def on_run_failed(run_id: str, diagnosis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Maybe turn a failed run into a repair proposal. Returns the proposal, or ``None`` and why (logged)."""
    if not enabled():
        return None
    from agent.agent_store import get_agent_store
    from agent.flow_revision import compute_flow_revision
    from agent.run_briefing import build_run_briefing

    store = get_agent_store()
    run = store.get_run(run_id)
    if not run or run.get("status") != "failed":
        return None
    agent_id = str(run.get("agent_id") or "")
    if not agent_id:
        return None
    klass = (diagnosis or {}).get("class")
    if klass in NOT_A_CONFIG_PROBLEM:
        logger.info("repair: run %s failed as %s — not a configuration problem, no proposal", run_id, klass)
        return None
    agent = store.get_agent(agent_id)
    if not agent or agent.get("is_pseudo"):
        return None
    proposals = get_repair_proposal_store()
    current_revision = compute_flow_revision(agent.get("flow_json"))
    proposals.supersede_stale(agent_id, current_revision)
    open_proposal = proposals.open_for_agent(agent_id)
    if open_proposal:
        proposals.add_seen_run(open_proposal["id"], run_id)
        logger.info("repair: run %s joins open proposal %s", run_id, open_proposal["id"])
        return open_proposal
    if proposals.created_today(agent_id) >= proposals_per_day():
        logger.warning("repair: daily proposal cap reached for agent %s; run %s not proposed", agent_id, run_id)
        return None

    briefing = build_run_briefing(agent_id)
    rejected = proposals.last_rejected(agent_id)
    if rejected:
        briefing = dict(briefing)
        briefing["previous_rejected_proposal"] = {"summary": rejected.get("summary"), "reason": rejected.get("reject_reason")}
    diagnosis = diagnosis or {}
    kind_hint = str((diagnosis.get("suggested_change") or {}).get("kind") or "")

    flow_json = None
    summary = ""
    warnings: list[Any] = []
    generated: list[Any] = []
    error: str | None = None
    if kind_hint not in (KIND_SCRIPT_CHANGE, KIND_ENVIRONMENT):
        try:
            generated, dropped, _assessment = await _suggest(agent, briefing, diagnosis)
            if not generated and dropped:
                error = "every proposal was dropped by the gate: " + "; ".join(str(d) for d in dropped)[:400]
        except Exception as exc:
            error = f"proposal generation failed: {type(exc).__name__}: {exc}"
            logger.warning("repair: %s (agent %s, run %s)", error, agent_id, run_id)
    kind = classify_kind(diagnosis, briefing, generated)
    if generated:
        first = generated[0]
        flow_json = first.flow
        summary = str(first.summary or "")
        warnings = [w.to_dict() if hasattr(w, "to_dict") else w for w in (getattr(first, "warnings", []) or [])]
    elif error:
        kind = KIND_UNKNOWN
        summary = error
    else:
        summary = str(diagnosis.get("cause") or "") or f"failed at {(briefing.get('runs') or [{}])[0].get('failed_step', {}).get('title', '?')}"
    proposal = proposals.create(
        agent_id=agent_id, run_id=run_id, base_flow_revision=current_revision,
        briefing=briefing, diagnosis=diagnosis, kind=kind, summary=summary,
        flow_json=flow_json, human_actions=human_actions_for(kind, diagnosis, briefing), warnings=warnings,
    )
    _notify(agent, proposal)
    return proposal


_KIND_LABEL = {
    KIND_WORKFLOW: "워크플로 변경", KIND_GENERATED_SCRIPT: "생성 스크립트", KIND_SCRIPT_CHANGE: "스크립트 수정 필요",
    KIND_ENVIRONMENT: "환경", KIND_UNKNOWN: "미분류",
}


def _notify(agent: dict[str, Any], proposal: dict[str, Any]) -> None:
    try:
        from agent.notification_store import get_notification_store

        kind = proposal.get("kind") or KIND_UNKNOWN
        if kind in APPLICABLE_KINDS:
            action = "대시보드/앱에서 제안을 검토하고 적용하세요."
        else:
            actions = proposal.get("human_actions") or []
            action = "사람이 해야 할 일: " + (actions[0].get("what") if actions else proposal.get("summary") or "")
        get_notification_store().create(
            title=f"{agent.get('name') or agent.get('id')} — 수리 제안 ({_KIND_LABEL.get(kind, kind)})",
            body=f"{proposal.get('summary') or ''}\n{action}\nproposal:{proposal['id']}",
            level="warning", run_id=proposal.get("run_id"), agent_id=agent.get("id"), reason=NOTIFICATION_REASON,
        )
    except Exception:
        logger.exception("repair: notification failed for proposal %s", proposal.get("id"))


__all__ = [
    "RepairProposalStore", "get_repair_proposal_store", "on_run_failed", "classify_kind",
    "human_actions_for", "STATUS_PROPOSED", "STATUS_ACCEPTED", "STATUS_REJECTED", "STATUS_SUPERSEDED",
    "STATUS_EXPIRED", "KINDS", "APPLICABLE_KINDS", "NOTIFICATION_REASON",
]
