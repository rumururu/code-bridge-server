"""The failure briefing: what an agent's recent runs did, as facts (AGENT_SELF_REPAIR_SPEC §1).

Every Configurator entry point receives this without the caller asking —
the canvas `suggest`, the revision conversation, and the phone's builder
through them. Before it existed the Configurator's inputs were the text a
person typed and the current flow: an agent that had failed six cycles in a
row, with a diagnosis step that had written the cause each time, reached the
Configurator as a blank intent box.

Facts only. Exit codes, output tails, what the diagnosis step wrote, how many
failures in a row — things read straight off the run record. Interpretation
is the diagnosis's job (§3), and the diagnosis cites its evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audit.audit_redactor import redact_payload

logger = logging.getLogger(__name__)

#: Most recent runs a briefing looks at.
DEFAULT_RUN_LIMIT = 5
#: Output kept per failed step (lines and bytes), after masking.
OUTPUT_TAIL_LINES = 40
OUTPUT_TAIL_BYTES = 2048
#: Whole-briefing budget; oldest runs are dropped first.
BRIEFING_BYTES = 12 * 1024

_INTERRUPTED_MARKER = "Interrupted: the server stopped"
_TERMINAL_FAILURE = {"failed"}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _tail(text: str) -> str:
    lines = text.rstrip("\n").split("\n")[-OUTPUT_TAIL_LINES:]
    joined = "\n".join(lines)
    if len(joined.encode("utf-8")) > OUTPUT_TAIL_BYTES:
        joined = joined.encode("utf-8")[-OUTPUT_TAIL_BYTES:].decode("utf-8", errors="ignore")
    return joined


def _masked(text: str | None) -> str | None:
    if text is None:
        return None
    redacted, _categories = redact_payload(text)
    return redacted if isinstance(redacted, str) else str(redacted)


def _step_brief(step: dict[str, Any]) -> dict[str, Any]:
    """One step, reduced to what a reader of a failure needs."""
    output = step.get("output") if isinstance(step.get("output"), dict) else {}
    shell = output.get("shell") if isinstance(output.get("shell"), dict) else {}
    error = output.get("error") if isinstance(output.get("error"), dict) else {}
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    tail_source = _text(shell.get("stderr")) or _text(shell.get("stdout"))
    result = output.get("result")
    brief: dict[str, Any] = {
        "step_id": step_input.get("workflow_step_id") or step.get("id"),
        "title": step.get("title"),
        "type": step_input.get("workflow_type") or step_input.get("type"),
        "sequence": step.get("sequence"),
        "status": step.get("status"),
    }
    if shell:
        brief["exit_code"] = shell.get("exit_code")
        brief["timed_out"] = bool(shell.get("timed_out"))
        if shell.get("contention"):
            brief["contention"] = [str(c) for c in shell["contention"]][:8]
    if tail_source:
        brief["output_tail"] = _masked(_tail(tail_source))
    if _text(error.get("message")):
        brief["error_message"] = _masked(error["message"])
    if isinstance(result, str) and result.strip():
        # An llm step's answer (a user-authored diagnosis step lands here).
        brief["result"] = _masked(_tail(result))
    return brief


def _run_brief(store: Any, run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run.get("id"))
    task_id = run.get("task_id")
    steps = store.list_run_steps(run_id) if run_id else []
    failed_index = next(
        (i for i, s in enumerate(steps) if s.get("status") in _TERMINAL_FAILURE), None
    )
    interrupted = False
    for step in steps:
        output = step.get("output") if isinstance(step.get("output"), dict) else {}
        error = output.get("error") if isinstance(output.get("error"), dict) else {}
        if _INTERRUPTED_MARKER in str(error.get("message") or ""):
            interrupted = True
            break
    if not interrupted:
        try:
            for event in store.list_events(run_id, limit=50):
                app_event = event.get("app_event") if isinstance(event.get("app_event"), dict) else {}
                err = app_event.get("error") if isinstance(app_event.get("error"), dict) else {}
                if _INTERRUPTED_MARKER in str(err.get("message") or ""):
                    interrupted = True
                    break
        except Exception:  # events are a bonus, never the reason a briefing fails
            logger.debug("run briefing: events unavailable for %s", run_id, exc_info=True)
    brief: dict[str, Any] = {
        "run_id": run_id,
        "task_id": task_id,
        "status": run.get("status"),
        "started_at": run.get("started_at") or run.get("created_at"),
        "ended_at": run.get("ended_at"),
        "interrupted_by_shutdown": interrupted,
        "failed_step": _step_brief(steps[failed_index]) if failed_index is not None else None,
        "later_steps": [
            _step_brief(s) for s in (steps[failed_index + 1 :] if failed_index is not None else [])
            if s.get("status") not in ("queued", "skipped")
        ],
        "diagnosis": _diagnosis_event(store, run_id),
    }
    return brief


def _diagnosis_event(store: Any, run_id: str) -> dict[str, Any] | None:
    """The system diagnosis event (§3), when one has been recorded."""
    try:
        for event in reversed(store.list_events(run_id, limit=200)):
            if event.get("event_type") == "run.diagnosis":
                app_event = event.get("app_event")
                return app_event if isinstance(app_event, dict) else None
    except Exception:
        logger.debug("run briefing: diagnosis lookup failed for %s", run_id, exc_info=True)
    return None


def _registered_scripts_for(flow: Any, store: Any) -> list[dict[str, Any]]:
    """The scripts this flow names, each with where it comes from.

    ``origin`` is what §4 turns on: a script under the server's generated
    directory is the Configurator's to re-propose; anything else is the
    operator's file, and a proposal may only describe a change to it.
    """
    if not isinstance(flow, list):
        return []
    wanted = {
        str(s.get("script_id"))
        for s in flow
        if isinstance(s, dict) and _text(s.get("script_id"))
    }
    if not wanted:
        return []
    try:
        from agent.script_store import get_script_store
        from routes.scripts import _MANAGED_SCRIPTS_DIR as GENERATED_SCRIPTS_DIR
    except Exception:
        return [{"script_id": sid, "origin": "unknown"} for sid in sorted(wanted)]
    generated_root = str(Path(GENERATED_SCRIPTS_DIR).expanduser())
    out = []
    try:
        scripts = {s.get("id"): s for s in get_script_store().list_scripts(limit=200)}
    except Exception:
        scripts = {}
    for sid in sorted(wanted):
        script = scripts.get(sid) or {}
        path = _text(script.get("path")) or ""
        origin = "generated" if path.startswith(generated_root) else ("registered" if path else "unknown")
        out.append({"script_id": sid, "name": script.get("name"), "path": path or None, "origin": origin})
    return out


def build_run_briefing(agent_id: str, *, limit: int = DEFAULT_RUN_LIMIT) -> dict[str, Any]:
    """The briefing for one agent — reads the store, writes nothing.

    ``runs`` is empty when the agent has no finished run that failed; callers
    render nothing then, so a healthy agent's prompt is byte-for-byte what it
    was before briefings existed.
    """
    from agent.agent_store import get_agent_store

    store = get_agent_store()
    agent = store.get_agent(agent_id) or {}
    recent = store.list_runs(agent_id=agent_id, limit=max(limit * 3, limit))
    finished = [r for r in recent if r.get("status") in ("failed", "completed", "cancelled")]
    failed = [r for r in finished if r.get("status") in _TERMINAL_FAILURE][:limit]
    briefs = [_run_brief(store, run) for run in failed]

    streak = 0
    for run in finished:
        if run.get("status") not in _TERMINAL_FAILURE:
            break
        brief = next((b for b in briefs if b["run_id"] == run.get("id")), None)
        if brief is not None and brief["interrupted_by_shutdown"]:
            continue
        streak += 1

    same_step = False
    real_failures = [b for b in briefs if not b["interrupted_by_shutdown"] and b["failed_step"]]
    if len(real_failures) >= 2:
        same_step = real_failures[0]["failed_step"]["step_id"] == real_failures[1]["failed_step"]["step_id"]

    briefing: dict[str, Any] = {
        "agent_id": agent_id,
        "agent_name": agent.get("name"),
        "runs": briefs,
        "streak": streak,
        "same_step_as_previous": same_step,
        "registered_scripts": _registered_scripts_for(agent.get("flow_json"), store),
        "truncated": False,
    }
    # Size budget: drop the oldest runs until the briefing fits.
    import json

    while briefing["runs"] and len(json.dumps(briefing, ensure_ascii=False).encode("utf-8")) > BRIEFING_BYTES:
        briefing["runs"].pop()
        briefing["truncated"] = True
    return briefing


def failure_briefing_block(briefing: dict[str, Any] | None) -> str:
    """The briefing as a prompt block — empty when there is nothing to say."""
    if not briefing or not briefing.get("runs"):
        return ""
    lines = [
        "",
        "Recent failed runs of this agent (facts from the run record). A proposal",
        "must address a failure listed here; do not invent a cause the record",
        "does not show. A run marked interrupted_by_shutdown failed because the",
        "server restarted, not because of its configuration.",
        f"Consecutive failures: {briefing.get('streak', 0)}"
        + (" (same step as the previous failure)" if briefing.get("same_step_as_previous") else ""),
    ]
    for run in briefing["runs"]:
        head = f"- run {run['run_id']} [{run.get('status')}] ended {run.get('ended_at') or '?'}"
        if run.get("interrupted_by_shutdown"):
            head += " — interrupted_by_shutdown"
        lines.append(head)
        step = run.get("failed_step")
        if step:
            desc = f"    failed step: {step.get('step_id')} ({step.get('type')}) \"{step.get('title')}\""
            if "exit_code" in step:
                desc += f" exit={step.get('exit_code')}" + (" timed_out" if step.get("timed_out") else "")
            lines.append(desc)
            if step.get("error_message"):
                lines.append(f"    error: {step['error_message']}")
            if step.get("contention"):
                lines.append("    other processes were driving the same device when it failed:")
                lines.extend("      " + c for c in step["contention"])
            if step.get("output_tail"):
                lines.append("    output tail:")
                lines.extend("      " + l for l in str(step["output_tail"]).split("\n")[-12:])
        for later in run.get("later_steps") or []:
            if later.get("result"):
                lines.append(f"    later step {later.get('step_id')} ({later.get('status')}) wrote:")
                lines.extend("      " + l for l in str(later["result"]).split("\n")[:8])
        diagnosis = run.get("diagnosis")
        if isinstance(diagnosis, dict) and diagnosis.get("class"):
            lines.append(f"    diagnosis: class={diagnosis.get('class')}"
                         + (f" cause={diagnosis.get('cause')}" if diagnosis.get("cause") else ""))
    scripts = briefing.get("registered_scripts") or []
    external = [s for s in scripts if s.get("origin") == "registered"]
    if external:
        lines.append("Scripts this workflow runs that are managed OUTSIDE this server (you cannot")
        lines.append("edit their contents; if the fix is inside one, say so and name the file):")
        for s in external:
            lines.append(f"  - {s.get('script_id')}: {s.get('name')} — {s.get('path')}")
    if briefing.get("truncated"):
        lines.append("(older runs omitted for size)")
    lines.append("")
    return "\n".join(lines)


__all__ = ["build_run_briefing", "failure_briefing_block"]
