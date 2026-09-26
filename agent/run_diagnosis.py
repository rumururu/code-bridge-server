"""System diagnosis of a failed run (AGENT_SELF_REPAIR_SPEC §3).

Diagnosis used to be a workflow step a person added by hand — an llm step
after the work, writing "why" into memory. An agent whose author left it out
got an exit code; an agent whose flow jumped over it on success never ran
it. Diagnosis is an attribute of a run, so the server records one for every
failed run: a deterministic class first (no model, always), then — when the
class is something a model can reason about — one call that names a cause
and **quotes its evidence** from the record. A cause with no quoted line is
not trusted and not stored as a cause.

The result is a run event, ``run.diagnosis``, read by the dashboard, the
phone, the failure briefing (§1) and the repair trigger (§2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from agent.run_briefing import _run_brief, failure_briefing_block

logger = logging.getLogger(__name__)

EVENT_TYPE = "run.diagnosis"

CLASS_INTERRUPTED = "interrupted"
CLASS_SCRIPT_MISSING = "script_missing"
CLASS_SCRIPT_FAILED = "script_failed"
CLASS_TIMED_OUT = "timed_out"
CLASS_TARGET_UNRESOLVED = "target_unresolved"
CLASS_PROVIDER_ERROR = "provider_error"
CLASS_APPROVAL_DENIED = "approval_denied"
CLASS_APPROVAL_EXPIRED = "approval_expired"
CLASS_PARK_ABANDONED = "park_abandoned"
CLASS_DEVICE_CONTENTION = "device_contention"
CLASS_UNKNOWN = "unknown"

#: Classes a model is not asked about: the cause is already the class.
NO_LLM_CLASSES = frozenset(
    {CLASS_INTERRUPTED, CLASS_APPROVAL_DENIED, CLASS_APPROVAL_EXPIRED, CLASS_PARK_ABANDONED, CLASS_DEVICE_CONTENTION}
)

_SCRIPT_MISSING_RE = re.compile(r"No such file|not found: .*\.sh|script .* does not exist|command not found", re.I)
_TARGET_RE = re.compile(r"unresolved (browser|app) target|target .* could not be resolved|placeholder", re.I)
_PROVIDER_RE = re.compile(r"provider|session|quota|rate limit|overloaded|401|403|api key", re.I)
_DENIED_RE = re.compile(r"denied|rejected by (the )?user", re.I)
_EXPIRED_RE = re.compile(r"expired", re.I)
_ABANDONED_RE = re.compile(r"abandon|nobody answered", re.I)


def llm_diagnosis_enabled() -> bool:
    return os.environ.get("CODEBRIDGE_RUN_DIAGNOSIS_LLM", "1") not in ("0", "false", "no")


def classify(run_brief: dict[str, Any]) -> str:
    """The deterministic class of a failed run, from its briefing alone."""
    if run_brief.get("interrupted_by_shutdown"):
        return CLASS_INTERRUPTED
    step = run_brief.get("failed_step") or {}
    error = str(step.get("error_message") or "")
    tail = str(step.get("output_tail") or "")
    text = error + "\n" + tail
    if _DENIED_RE.search(error) and "approval" in error.lower():
        return CLASS_APPROVAL_DENIED
    if _EXPIRED_RE.search(error) and "approval" in error.lower():
        return CLASS_APPROVAL_EXPIRED
    if _ABANDONED_RE.search(error):
        return CLASS_PARK_ABANDONED
    if step.get("contention"):
        # Another process on this machine was driving the same device. The
        # script's own exit says nothing about a fight it did not know it
        # was in; this outranks timed_out and script_failed.
        return CLASS_DEVICE_CONTENTION
    if step.get("timed_out"):
        return CLASS_TIMED_OUT
    if "exit_code" in step:
        if _SCRIPT_MISSING_RE.search(text) and step.get("exit_code") in (127, 126, 2):
            return CLASS_SCRIPT_MISSING
        return CLASS_SCRIPT_FAILED
    if step.get("type") in ("browser_action", "app_action") and _TARGET_RE.search(text):
        return CLASS_TARGET_UNRESOLVED
    if step.get("type") == "llm" and _PROVIDER_RE.search(error):
        return CLASS_PROVIDER_ERROR
    return CLASS_UNKNOWN


def _prompt(run_brief: dict[str, Any], klass: str, agent_name: str) -> str:
    briefing = {"agent_name": agent_name, "runs": [run_brief], "streak": 1, "same_step_as_previous": False}
    return (
        "You diagnose one failed run of a Code Bridge agent. Answer from the record\n"
        "below only. Quote the line(s) that show the cause; if no line shows it,\n"
        "say so in `cause` and leave `evidence` empty — do not invent a cause.\n"
        f"Deterministic class already assigned: {klass}\n"
        + failure_briefing_block(briefing)
        + "\nRespond with ONE fenced ```json block:\n"
        "{\"cause\": \"one sentence\", \"evidence\": [\"quoted line\", ...],\n"
        " \"suggested_change\": {\"kind\": \"workflow|generated_script|script_change|environment|unknown\",\n"
        "                      \"what\": \"...\", \"where\": \"...\"},\n"
        " \"needs_human\": true|false}\n"
        "kind: workflow = the agent's steps/policies/args need changing; generated_script = a\n"
        "new script this server could generate would fix it; script_change = a script managed\n"
        "outside this server must change (name it); environment = the device, app, permissions\n"
        "or network must change (say where). No prose outside the block.\n"
    )


def _parse(text: str) -> dict[str, Any] | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    evidence = [e for e in (data.get("evidence") or []) if isinstance(e, str) and e.strip()]
    change = data.get("suggested_change") if isinstance(data.get("suggested_change"), dict) else {}
    kind = str(change.get("kind") or "unknown")
    if kind not in {"workflow", "generated_script", "script_change", "environment", "unknown"}:
        kind = "unknown"
    return {
        # A cause with nothing quoted is an opinion; keep the sentence but mark it.
        "cause": str(data.get("cause") or "").strip() or None,
        "evidence": evidence[:3],
        "unsupported": not evidence,
        "suggested_change": {"kind": kind, "what": str(change.get("what") or ""), "where": str(change.get("where") or "")},
        "needs_human": bool(data.get("needs_human")) or kind in {"script_change", "environment"},
    }


async def _one_shot(prompt: str, *, timeout: float = 90.0) -> str:
    """One model turn with no tools: text in, text out."""
    from chat.chat_session_service import create_chat_session, get_chat_provider_selection
    import uuid
    from pathlib import Path

    session = await create_chat_session(f"run-diagnosis-{uuid.uuid4().hex[:8]}", str(Path.cwd()), get_chat_provider_selection())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    chunks: list[str] = []
    try:
        stream = session.send_message(prompt)
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                event = await asyncio.wait_for(stream.__anext__(), remaining)
            except StopAsyncIteration:
                break
            kind = event.get("type")
            if kind == "result":
                result = event.get("result")
                if isinstance(result, str):
                    return result
                break
            if kind == "assistant":
                message = event.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    chunks.extend(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
                continue
            if kind == "control_request":
                # No tools here — a diagnosis reads the record it was given.
                break
            if kind == "error":
                err = event.get("error")
                raise RuntimeError((err or {}).get("message") if isinstance(err, dict) else "LLM error")
    finally:
        close = getattr(session, "close", None)
        if close is not None:
            try:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass
    return "".join(chunks)


async def diagnose_run(run_id: str, *, use_llm: bool | None = None) -> dict[str, Any] | None:
    """Diagnose one failed run and record ``run.diagnosis``. Once per run."""
    from agent.agent_store import get_agent_store

    store = get_agent_store()
    run = store.get_run(run_id)
    if not run or run.get("status") != "failed":
        return None
    for event in store.list_events(run_id, limit=200):
        if event.get("event_type") == EVENT_TYPE:
            return event.get("app_event") if isinstance(event.get("app_event"), dict) else None
    brief = _run_brief(store, run)
    klass = classify(brief)
    diagnosis: dict[str, Any] = {
        "class": klass,
        "failed_step_id": (brief.get("failed_step") or {}).get("step_id"),
        "cause": None,
        "evidence": [],
        "suggested_change": None,
        "needs_human": klass in {CLASS_APPROVAL_DENIED, CLASS_APPROVAL_EXPIRED, CLASS_PARK_ABANDONED},
        "llm": "skipped",
    }
    if klass == CLASS_INTERRUPTED:
        diagnosis["cause"] = "The server restarted while this run was in progress."
        diagnosis["suggested_change"] = {"kind": "unknown", "what": "nothing — the schedule re-fires after a restart", "where": ""}
    if klass == CLASS_DEVICE_CONTENTION:
        others = list((brief.get("failed_step") or {}).get("contention") or [])
        diagnosis["cause"] = "Another process on this machine was driving the same device while this step ran; the two automations fought over the screen."
        diagnosis["evidence"] = others[:3]
        diagnosis["suggested_change"] = {
            "kind": "environment",
            "what": "Stop the other automation that uses this device, or change one of the two schedules so they never overlap: " + (others[0] if others else ""),
            "where": "this Mac (process list at failure time)",
        }
        diagnosis["needs_human"] = True
    wants_llm = llm_diagnosis_enabled() if use_llm is None else use_llm
    if wants_llm and klass not in NO_LLM_CLASSES:
        agent = store.get_agent(str(run.get("agent_id") or "")) or {}
        try:
            text = await _one_shot(_prompt(brief, klass, str(agent.get("name") or run.get("agent_id") or "")))
            parsed = _parse(text)
            if parsed is None:
                diagnosis["llm"] = "unparseable"
                diagnosis["error"] = "model answer had no json block"
            else:
                diagnosis.update(parsed)
                diagnosis["llm"] = "ok"
        except Exception as exc:  # the class is still worth recording
            logger.warning("run diagnosis: model call failed for %s: %s", run_id, exc)
            diagnosis["llm"] = "failed"
            diagnosis["error"] = f"{type(exc).__name__}: {exc}"
    store.append_event(run_id=run_id, event_type=EVENT_TYPE, app_event=diagnosis)
    return diagnosis


_BACKGROUND: set[asyncio.Task[Any]] = set()


def schedule_diagnosis(run_id: str) -> None:
    """Diagnose in the background from sync code; never blocks the caller."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        # No loop on this thread (a sync caller off the event loop): record
        # the deterministic class now; the model pass is skipped.
        try:
            asyncio.run(diagnose_run(run_id, use_llm=False))
        except Exception:
            logger.exception("run diagnosis: synchronous fallback failed for %s", run_id)
        return
    task = loop.create_task(_diagnose_then_repair(run_id))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


async def _diagnose_then_repair(run_id: str) -> None:
    """Diagnosis first, then the repair trigger (§2) — each best-effort."""
    diagnosis = None
    try:
        diagnosis = await diagnose_run(run_id)
    except Exception:
        logger.exception("run diagnosis failed for %s", run_id)
    try:
        from agent.repair_proposals import on_run_failed

        await on_run_failed(run_id, diagnosis)
    except Exception:
        logger.exception("repair trigger failed for %s", run_id)


__all__ = ["diagnose_run", "schedule_diagnosis", "classify", "EVENT_TYPE"]
