"""Read models for the phone's agent overview, work queue, and run history."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from datetime import datetime, timezone
from typing import Any

from agent.agent_store import get_agent_store
from agent.repair_proposals import get_repair_proposal_store
from agent.schedule_store import get_schedule_store
from approvals.approval_store import get_approval_store
from core.database import get_db_connection
from core.timestamps import to_utc_iso
from agent.agent_store import _row_to_run_record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cursor(key: tuple[Any, ...]) -> str:
    return base64.urlsafe_b64encode(json.dumps(key).encode()).decode().rstrip("=")


def _decode_cursor(value: str | None) -> tuple[str, int] | None:
    if not value:
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
        if not isinstance(data, list) or len(data) != 2:
            raise ValueError
        return str(data[0]), int(data[1])
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error) as exc:
        raise ValueError("invalid cursor") from exc


def _decode_action_cursor(value: str | None) -> tuple[int, str, str] | None:
    if not value:
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
        if not isinstance(data, list) or len(data) != 3:
            raise ValueError
        return int(data[0]), str(data[1]), str(data[2])
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error) as exc:
        raise ValueError("invalid cursor") from exc


def history(*, project_name: str | None, status: str | None, limit: int, cursor: str | None,
            since: str | None = None, until: str | None = None) -> dict[str, Any]:
    key = _decode_cursor(cursor)
    conditions: list[str] = []
    args: list[Any] = []
    if project_name:
        conditions.append("project_name = ?")
        args.append(project_name)
    if status:
        conditions.append("status = ?")
        args.append(status)
    if since:
        conditions.append("date(created_at) >= date(?)")
        args.append(since)
    if until:
        conditions.append("date(created_at) <= date(?)")
        args.append(until)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    page_where = where
    page_args = list(args)
    if key:
        page_where += (" AND " if conditions else " WHERE ") + "(created_at, rowid) < (?, ?)"
        page_args.extend(key)
    with get_db_connection(use_row_factory=True) as conn:
        total = conn.execute("SELECT COUNT(*) FROM agent_runs" + where, args).fetchone()[0]
        rows = conn.execute(
            "SELECT rowid AS page_rowid, * FROM agent_runs" + page_where
            + " ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (*page_args, limit + 1),
        ).fetchall()
    visible = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = visible[-1]
        next_cursor = _cursor((last["created_at"], last["page_rowid"]))
    return {"runs": [_row_to_run_record(row) for row in visible], "total_count": total, "next_cursor": next_cursor}


def action_items(*, limit: int, cursor: str | None) -> dict[str, Any]:
    from agent.approval_resume import is_settling_run

    store = get_agent_store()
    items: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()
    pending_approvals = get_approval_store().list_pending()
    pending_ids = {approval["id"] for approval in pending_approvals}
    for approval in pending_approvals:
        run_id = approval.get("run_id")
        context = store.get_run_checkpoint(run_id) if run_id else None
        checkpoint = context.get("checkpoint") if context else None
        matching = isinstance(checkpoint, dict) and checkpoint.get("approval_id") == approval["id"]
        run = context.get("run") if matching else (store.get_run(run_id) if run_id else None)
        step = context.get("step") if matching else None
        if matching and step:
            covered.add((run_id, step["id"]))
        items.append({"id": approval["id"], "kind": "approval", "title": approval.get("operation") or "Approval",
                      "run_id": run_id, "task_id": run.get("task_id") if run else None,
                      "agent_id": run.get("agent_id") if run else None, "approval_id": approval["id"],
                      "proposal_id": None, "created_at": approval["created_at"], "status": "pending",
                      "details": {"risk_level": approval.get("risk_level"), "desktop_only": approval.get("desktop_only")}})
    with get_db_connection(use_row_factory=True) as conn:
        waiting = conn.execute("""SELECT s.id AS step_id, s.run_id, s.task_id, s.updated_at,
            s.output_json, t.assigned_agent_id FROM agent_task_steps s
            JOIN agent_tasks t ON t.id = s.task_id
            WHERE s.status IN ('waiting_for_user', 'blocked')""").fetchall()
    for row in waiting:
        if not row["run_id"] or (row["run_id"], row["step_id"]) in covered:
            continue
        output = json.loads(row["output_json"] or "{}")
        checkpoint = output.get("checkpoint") if isinstance(output, dict) else None
        if not isinstance(checkpoint, dict):
            continue
        approval_id = checkpoint.get("approval_id")
        if approval_id in pending_ids:
            continue
        if approval_id and get_approval_store().get_request(approval_id):
            decision = get_approval_store().get_latest_decision(approval_id)
            if not decision:
                continue
            kind, status = "recovery", "pending" if is_settling_run(row["run_id"]) else "recovery_required"
        else:
            kind, status = "checkpoint", "waiting_for_user"
        items.append({"id": row["step_id"], "kind": kind, "title": checkpoint.get("prompt") or "Response needed",
                      "run_id": row["run_id"], "task_id": row["task_id"], "agent_id": row["assigned_agent_id"],
                      "approval_id": approval_id, "proposal_id": None, "created_at": to_utc_iso(row["updated_at"]),
                      "status": status, "details": {"reason": checkpoint.get("reason"), "step_id": row["step_id"]}})
    for proposal in get_repair_proposal_store().list_open():
        items.append({"id": proposal["id"], "kind": "repair_proposal", "title": proposal.get("summary") or "Repair proposal",
                      "run_id": proposal.get("run_id"), "task_id": None, "agent_id": proposal.get("agent_id"),
                      "approval_id": None, "proposal_id": proposal["id"], "created_at": proposal["created_at"],
                      "status": proposal["status"], "details": {"kind": proposal.get("kind")}})
    priority = {"approval": 0, "recovery": 0, "checkpoint": 1, "repair_proposal": 2}
    items.sort(key=lambda item: (priority[item["kind"]], item["created_at"] or "", item["id"]))
    total = len(items)
    key = _decode_action_cursor(cursor)
    if key:
        items = [item for item in items if (priority[item["kind"]], item["created_at"] or "", item["id"]) > key]
    page = items[:limit]
    return {"items": page, "total_count": total,
            "next_cursor": _cursor((priority[page[-1]["kind"]], page[-1]["created_at"] or "", page[-1]["id"]))
            if len(items) > limit else None, "as_of": _now()}


def overview() -> dict[str, Any]:
    store = get_agent_store()
    errors: list[str] = []

    def read_section(name, read, fallback):
        try:
            return read()
        except Exception:
            logging.getLogger(__name__).exception("Overview section unavailable: %s", name)
            errors.append(name)
            return fallback

    running = read_section("running", lambda: store.list_runs(statuses=("queued", "starting", "running"), limit=20), [])
    recent = read_section("recent", lambda: store.list_runs(statuses=("completed", "failed", "cancelled"), limit=20), [])
    schedules = read_section("next_schedules", lambda: [s for s in get_schedule_store().list_all(enabled_only=True) if s.get("next_run_at")], [])
    schedules.sort(key=lambda schedule: schedule["next_run_at"])
    count = read_section("action_count", lambda: action_items(limit=1, cursor=None)["total_count"], None)
    return {"running": running, "recent": recent, "action_count": count,
            "next_schedules": schedules[:10], "as_of": _now(), "section_errors": errors}


def get_review(run_id: str) -> dict[str, Any]:
    with get_db_connection(use_row_factory=True) as conn:
        row = conn.execute("SELECT * FROM agent_run_reviews WHERE run_id = ?", (run_id,)).fetchone()
    return {"reviewed": bool(row["reviewed"]) if row else False,
            "reviewed_at": to_utc_iso(row["reviewed_at"]) if row else None,
            "actor": row["actor"] if row else None}


def set_review(run_id: str, reviewed: bool, actor: str) -> dict[str, Any]:
    with get_db_connection() as conn:
        conn.execute("""INSERT INTO agent_run_reviews (run_id, reviewed, reviewed_at, actor)
            VALUES (?, ?, CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END, ?)
            ON CONFLICT(run_id) DO UPDATE SET reviewed = excluded.reviewed,
            reviewed_at = excluded.reviewed_at, actor = excluded.actor""",
            (run_id, int(reviewed), int(reviewed), actor if reviewed else None))
        conn.commit()
    return get_review(run_id)


def summary(run_id: str) -> dict[str, Any]:
    store = get_agent_store()
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    with get_db_connection(use_row_factory=True) as conn:
        rows = conn.execute("""SELECT e.*, p.event_id AS trusted_preflight FROM agent_events e
            LEFT JOIN agent_preflight_evidence p ON p.event_id = e.id AND p.run_id = e.run_id
            WHERE e.run_id = ? AND e.event_type IN ('preflight.completed', 'task.execution.failed', 'schedule.refired', 'schedule.refire_origin')
            ORDER BY e.sequence""", (run_id,)).fetchall()
    checks: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    interruption = None
    related_runs: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["app_event_json"] or "{}")
        if row["event_type"] in ("schedule.refired", "schedule.refire_origin") and payload.get("run_id"):
            related_runs.append({"run_id": payload["run_id"], "relation": row["event_type"], "event_id": row["id"]})
        if row["event_type"] == "preflight.completed" and row["trusted_preflight"]:
            checks = [{"command": result.get("command"), "passed": result.get("passed")}
                      for result in payload.get("results", []) if isinstance(result, dict)]
            evidence.append({"event_id": row["id"], "type": row["event_type"]})
        elif "Interrupted: the server stopped" in str(payload.get("error", {}).get("message", "")):
            interruption = {"reason": "server_stopped", "event_id": row["id"]}
    verification_status = "unknown" if not checks else ("passed" if all(c["passed"] is True for c in checks) else "failed")
    section_errors: list[str] = []
    try:
        artifacts = [{key: a.get(key) for key in ("id", "kind", "path", "mime_type", "metadata", "created_at")}
                     for a in store.list_artifacts(run_id)]
    except Exception:
        logging.getLogger(__name__).exception("Run summary section unavailable: artifacts")
        artifacts = None
        section_errors.append("artifacts")
    return {"run": run, "verification": {"status": verification_status, "checks": checks, "evidence": evidence},
            "artifacts": artifacts, "review": get_review(run_id), "interruption": interruption,
            "related_runs": related_runs, "section_errors": section_errors}
