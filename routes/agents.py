"""Agent Cockpit API routes."""

import asyncio
import logging
import re
from contextlib import suppress

logger = logging.getLogger(__name__)

# Keep strong references to background tasks created with
# `asyncio.create_task`. Without this set the event loop only holds a weak
# reference, so a fire-and-forget orchestrator turn can be garbage collected
# mid-run.
_BACKGROUND_TASKS: set = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from agent.configurator import (
    BuilderSession,
    build_configurator_system_prompt,
    build_configurator_turn_prompt,
    create_builder_session,
    delete_builder_session,
    get_builder_session,
    looks_like_manual_timing,
    resolve_task_draft_workdir,
    task_goal_from_draft,
)
from agent.agent_models import (
    AgentCreate,
    AgentArtifactCreate,
    AgentConnectorRequestUpdate,
    AgentDraft,
    AgentEventCreate,
    AgentUpdate,
    AgentRunOnceRequest,
    BuilderCommitRequest,
    BuilderStepDebugRequest,
    BuilderTurn,
    BuilderTurnResponse,
    DryRunRequest,
    MemoryCreate,
    MemoryUpdate,
    AgentRunCreate,
    AgentRunMessageCreate,
    AgentRunPreflightCreate,
    AgentTaskCreate,
    AgentTaskRunLinkCreate,
    AgentTaskStepRespondCreate,
    AgentTaskStepRunCreate,
    AgentTaskStepUpdate,
    AgentTaskStartCreate,
    AgentTaskUpdate,
    BrowserHandoffActionCreate,
    TaskDraft,
)
from agent.agent_origin import (
    AgentPromptNotEditableError,
    assert_patch_reaches_execution,
    resolve_agent_origin,
)
from agent.agent_store import (
    AgentStoreConflictError,
    PseudoAgentProtectedError,
    add_memory_from_event,
    get_agent_store,
)
from agent.browser_action_adapter import (
    get_browser_runtime_readiness,
    get_cached_browser_readiness_sync,
)
from agent.browser_session_store import get_browser_session_store
from agent.capability_registry import refresh_capability_registry
from agent.schedule_store import compute_next_fire_at, get_schedule_store
from agent.scheduler import get_scheduler
from agent.task_orchestrator import (
    complete_connector_request,
    execute_task_orchestration,
    execute_task_step_adapter,
    prepare_task_orchestration,
    resume_task_orchestration,
)
from agent.browser_action_executor import execute_browser_actions
from agent.tool_artifacts import ARTIFACT_ROOT, record_tool_action_result
from agent.workflow_contract import (
    BROWSER_STEP_TYPES,
    CODE_BROWSER_RUNTIME_UNAVAILABLE,
    CODE_UNKNOWN_STEP_REFERENCE,
    CODE_UNRESOLVED_BROWSER_TARGET,
    ContractReport,
    analyze_workflow,
)
from agent.workflow_v2 import WorkflowNormalizationError, normalize_workflow
from agent.workflow_step_schema import get_step_schema as get_workflow_step_schema_payload
from approvals.approval_service import decide_approval
from approvals.approval_store import get_approval_store
from chat.chat_session_service import create_chat_session, get_chat_provider_selection
from audit.route_audit import record_api_action
from core.database import get_project_db
from policy.policy_gate import evaluate_direct_action_gate
from terminal_action_service import execute_terminal_command_for_current_server

from .deps import verify_api_key
from .script_proposals import (
    draft_pending_proposals,
    list_proposals_for_session,
    proposal_view,
    sync_proposals_for_session,
)

router = APIRouter(prefix="/api/agent", tags=["agent"])

PSEUDO_AGENT_IDS = {"agent_legacy_chat", "agent_adhoc_dev"}
BUILDER_CONVERSE_FAST_TIMEOUT_SECONDS = 20.0
BUILDER_CONVERSE_JOB_TTL = timedelta(minutes=30)


@dataclass
class BuilderConverseJob:
    id: str
    session_id: str
    user_message: str
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    response: BuilderTurnResponse | None = None
    error: str | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)


BUILDER_CONVERSE_JOBS: dict[str, BuilderConverseJob] = {}


def _store():
    return get_agent_store()


def _require_run(run_id: str) -> dict[str, Any]:
    run = _store().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return run


def _require_artifact(run_id: str, artifact_id: str) -> dict[str, Any]:
    _require_run(run_id)
    artifact = _store().get_artifact(artifact_id)
    if not artifact or artifact.get("run_id") != run_id:
        raise HTTPException(status_code=404, detail=f"Agent artifact '{artifact_id}' not found")
    return artifact


def _require_agent(agent_id: str) -> dict[str, Any]:
    agent = _store().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent


def _with_script_names(flow: Any) -> Any:
    """Annotate shell steps with the name of the script they run.

    The stored definition only holds ``script_id``. Clients showing a workflow
    would otherwise render "step 1" with no hint that it shells out, or print a
    raw id at the user — neither says what is about to run on their machine.
    """
    if not isinstance(flow, list):
        return flow
    ids = {
        step.get("script_id")
        for step in flow
        if isinstance(step, dict) and step.get("type") == "shell" and step.get("script_id")
    }
    if not ids:
        return flow
    from agent.script_store import get_script_store

    store = get_script_store()
    names: dict[str, dict[str, Any]] = {}
    for script_id in ids:
        script = store.get(str(script_id))
        if script:
            names[str(script_id)] = script
    annotated = []
    for step in flow:
        if isinstance(step, dict) and step.get("type") == "shell":
            script = names.get(str(step.get("script_id")))
            step = {
                **step,
                "script_name": script["name"] if script else None,
                "script_path": script["path"] if script else None,
            }
        annotated.append(step)
    return annotated


def _flow_graph_view(flow_json: Any) -> dict[str, Any]:
    """The kernel graph view of a stored workflow — or the reason there is none.

    Returns exactly one key: ``flow_graph`` with the derived graph, or
    ``flow_graph_unavailable`` with a machine-readable ``reason`` and a
    sentence a person can act on. Never both, and never neither: a client
    that finds no graph can always say *why* it found none. Dropping the
    field silently (or sending ``flow_graph: null``) is the failure this
    project keeps having to undo — an absent field reads as "this agent has
    no graph", which is a claim about the agent when the truth is a claim
    about this server.

    ``agent.flow_graph`` is imported **inside this function on purpose**. It
    imports ``agent_flow_core`` at module top, and the deployed server venv
    (``~/.code-bridge/venv``) does not have that kernel installed yet. A
    module-level import here would therefore not degrade the graph field —
    it would stop the whole server from starting, taking every route with
    it. The ``except`` is narrowed to ``ImportError`` so a genuine bug
    inside the converter still surfaces as a bug.

    The derivation runs on the *stored* ``flow_json``, not the
    ``_with_script_names`` annotated copy: the graph is a view of the canon,
    and ``script_name``/``script_path`` are display sugar this route adds,
    not fields the workflow contract knows about.
    """

    try:
        from agent.flow_graph import UnsupportedTopologyError, to_graph
    except ImportError as exc:  # kernel absent — the deployed-venv case
        return {
            "flow_graph_unavailable": {
                "reason": "kernel_not_installed",
                "message": (
                    "This server's Python environment has no agent-flow-core"
                    " kernel, so the graph view cannot be derived. The"
                    " workflow itself is unaffected — flow_json is the canon"
                    " and runs exactly as stored."
                ),
                "detail": str(exc),
            }
        }

    try:
        normalized = normalize_workflow(flow_json)
    except WorkflowNormalizationError as exc:
        # A stored workflow older than a normalization rule, or written
        # straight into the store. It cannot run either, so saying so here is
        # the useful answer, not a 500.
        return {
            "flow_graph_unavailable": {
                "reason": "not_normalizable",
                "message": (
                    "The stored workflow does not normalize, so no graph can"
                    f" be derived from it: {exc}"
                ),
                "detail": str(exc),
            }
        }

    try:
        flow = to_graph(normalized)
    except UnsupportedTopologyError as exc:
        return {
            "flow_graph_unavailable": {
                "reason": "not_linear",
                "message": str(exc),
                "issues": exc.detail["issues"],
            }
        }
    except Exception as exc:
        # Broad on purpose, and only here: a derived convenience view must
        # not be able to take down the read of an agent. It is *reported*,
        # not swallowed — the reason and the exception text both ride out.
        logger.exception("flow_graph derivation failed")
        return {
            "flow_graph_unavailable": {
                "reason": "conversion_failed",
                "message": (
                    "Deriving the graph view raised an unexpected error;"
                    " the stored workflow is unchanged."
                ),
                "detail": f"{type(exc).__name__}: {exc}",
            }
        }

    return {"flow_graph": flow.model_dump(by_alias=True)}


_ACTIVE_RUN_STATUSES = ("queued", "starting", "running")

# A run in one of these has stopped and is waiting on the user. Counting it as
# "active" is what made an agent that needs an answer read as busy: the list
# said "활성 1개", which is the same thing it says while a run is working fine,
# so the one agent that needed the user was the one they had no reason to open.
# `agent/scheduler.py::_WAITING_RUN_STATUSES` is the same set for the same runs.
_WAITING_RUN_STATUSES = ("blocked", "waiting_for_user", "waiting_user")


def _agent_run_activity(agent_id: str) -> dict[str, Any]:
    """When this agent last ran, how it went, and what it is doing now.

    The client renders "no runs yet" from ``last_fire_at``, so an agent that
    has run sixty times still reads as never-used until this is filled in —
    the run history existed, it just never reached the phone. ``last_run_status``
    is the other half: a list that shows only *when* cannot distinguish sixty
    clean runs from sixty failures.

    ``last_run_status`` describes the same run ``last_fire_at`` names, so the
    two can never be read as one sentence about two different runs.
    """
    empty = {
        "last_fire_at": None,
        "last_run_status": None,
        "active_run_count": 0,
        "waiting_run_count": 0,
    }
    store = _store()
    try:
        runs = store.list_runs(agent_id=agent_id, limit=50)
    except Exception:
        return dict(empty)
    if not runs:
        return dict(empty)

    latest: dict[str, Any] | None = None
    latest_stamp = ""
    for run in runs:
        stamp = str(run.get("started_at") or run.get("created_at") or "")
        if not stamp:
            continue
        # Strict `>`: `list_runs` already returns newest-first, so a tie on
        # SQLite's second-resolution timestamps keeps the newer row.
        if latest is None or stamp > latest_stamp:
            latest = run
            latest_stamp = stamp

    last_run = latest if latest is not None else runs[0]
    return {
        "last_fire_at": _as_utc_iso(latest_stamp) if latest_stamp else None,
        "last_run_status": str(last_run.get("status") or "") or None,
        "active_run_count": sum(
            1 for run in runs if run.get("status") in _ACTIVE_RUN_STATUSES
        ),
        "waiting_run_count": sum(
            1 for run in runs if run.get("status") in _WAITING_RUN_STATUSES
        ),
    }


def _as_utc_iso(stamp: str) -> str | None:
    """Turn a stored timestamp into an unambiguous instant.

    SQLite writes ``CURRENT_TIMESTAMP`` as naive UTC — "2026-08-01 02:57:48"
    with nothing to say which zone that is. Sent as-is, a client in KST reads
    it as local and puts the run nine hours in the past. ``next_fire_at``
    already goes out with an offset, so the two fields disagreed about what
    "now" meant.
    """
    text = stamp.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _agent_with_next_fire(
    agent: dict[str, Any],
    *,
    include_flow_graph: bool = False,
) -> dict[str, Any]:
    """The stored agent plus everything a client needs that is not a column.

    ``origin`` is the one that changes what the client is allowed to *offer*.
    Without it an agent registered from a Claude Code agent file looks exactly
    like one authored here, so every client hands the user a prompt box for
    text that will never execute — see :mod:`agent.agent_origin`.

    ``include_flow_graph`` is off by default and turned on by exactly one
    caller: ``GET /agents/{agent_id}``. The graph is a per-step derivation
    (a kernel ``FlowStep`` model built for every step, plus the edge
    derivation), so on the list route — which serves up to 200 agents in one
    response — the cost multiplies by the agent count for a field a list view
    has no use for: a list draws names, schedules and run status, and opening
    an agent is what asks about its shape. Write responses (create/patch/
    builder-commit) share this serializer and stay off it too: they echo what
    was just saved, and the caller that wants the derived view reads the agent
    back. Turning it on for a caller is one keyword away if that changes.
    """
    agent_id = str(agent["id"])
    next_fire = compute_next_fire_at(agent_id)
    activation = _store().get_agent_activation_summary(agent_id)
    payload = {
        **agent,
        "flow_json": _with_script_names(agent.get("flow_json")),
        "next_fire_at": next_fire.isoformat() if next_fire else None,
        **_agent_run_activity(agent_id),
        "activation": activation,
        "origin": resolve_agent_origin(agent_id).to_view(),
    }
    if include_flow_graph:
        payload.update(_flow_graph_view(agent.get("flow_json")))
    return payload


def _resolve_agent_task(agent_id: str, task_id: str | None) -> dict[str, Any] | None:
    """Find the task this agent's next run — real or simulated — would use.

    An explicit ``task_id`` is validated against the agent (404 if it does not
    exist, 409 if it belongs to someone else). With none given, the agent's
    first assigned task stands in: most agents have exactly one, and asking
    the user to name it every time they just want to try the thing they built
    would be busywork. Returns ``None`` when the agent has no task at all —
    callers decide what that means for them (dry-run still has something to
    preview; a real run does not, and says so).
    """
    store = _store()
    if task_id:
        task = store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
        if task.get("assigned_agent_id") != agent_id:
            raise HTTPException(
                status_code=409,
                detail="Task is not assigned to this agent",
            )
        return task

    for task in store.list_tasks(limit=200):
        if task.get("assigned_agent_id") == agent_id:
            return task
    return None


def _task_cwd(task: dict[str, Any] | None) -> str | None:
    if not isinstance(task, dict):
        return None
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        cwd = metadata.get("cwd") or metadata.get("project_path") or metadata.get("root_path")
        if isinstance(cwd, str) and cwd.strip():
            return cwd.strip()
    return None


def _schedule_expression_from_draft(schedule: str | None) -> dict[str, Any] | None:
    """Turn the Configurator's schedule phrase into a schedule expression.

    The phrase is whatever the model wrote down, in whatever language the user
    was speaking. Anything this cannot read is dropped silently by the caller,
    so a Korean user asking for "매일 아침 9시" got an agent with no schedule
    and nothing saying why — which is the same silent nothing that "paired but
    never runs" looks like from the outside.
    """
    if not isinstance(schedule, str):
        return None
    text = schedule.strip().lower()
    if not text or text in {"now", "manual", "none", "no schedule"}:
        return None
    if any(word in text for word in ("수동", "직접 실행", "안 함", "없음")):
        return None
    import re

    _EN_UNITS = {
        "minute": 60, "minutes": 60, "min": 60, "m": 60,
        "hour": 3600, "hours": 3600, "hr": 3600, "h": 3600,
        "day": 86400, "days": 86400, "d": 86400,
    }
    _KO_UNITS = {"분": 60, "시간": 3600, "일": 86400}

    def _interval(seconds: int) -> dict[str, Any]:
        # A sub-minute schedule is a runaway loop, not a schedule.
        return {"kind": "interval", "seconds": max(seconds, 60)}

    interval = re.search(
        r"every\s+(\d+)\s*(minute|minutes|min|m|hour|hours|hr|h|day|days|d)\b",
        text,
    )
    if interval:
        return _interval(int(interval.group(1)) * _EN_UNITS[interval.group(2)])

    compact = re.fullmatch(r"every\s+(\d+)([mhd])", text)
    if compact:
        return _interval(int(compact.group(1)) * _EN_UNITS[compact.group(2)])

    # Korean puts the interval either before ("매 6시간") or after ("6시간마다",
    # "6시간에 한 번"). Both are ordinary ways to say it and neither is a typo.
    korean_interval = re.search(
        r"(?:매\s*(\d+)\s*(분|시간|일)"
        r"|(\d+)\s*(분|시간|일)\s*(?:마다|간격|에\s*한\s*번|당\s*한\s*번))",
        text,
    )
    if korean_interval:
        amount = korean_interval.group(1) or korean_interval.group(3)
        unit = korean_interval.group(2) or korean_interval.group(4)
        return _interval(int(amount) * _KO_UNITS[unit])

    _MORNING = {"오전", "아침", "새벽", "am"}
    _AFTERNOON = {"오후", "저녁", "밤", "pm"}

    def _daily(hour: int, minute: int, period: str | None) -> dict[str, Any] | None:
        # 오후 12시 is midday and 오전 12시 is midnight; adding 12 to both would
        # put one of them a full half-day out.
        if period in _AFTERNOON and hour < 12:
            hour += 12
        elif period in _MORNING and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            return None
        return {"kind": "daily_at", "time": f"{hour:02d}:{minute:02d}"}

    # "daily at 9am", "every day at 9:30 pm" — and the same said back to front.
    # Checked before the 24-hour form, which would otherwise read the "9" of
    # "9:30 pm" as 09:30 and schedule the run twelve hours early.
    daily_meridiem = re.search(
        r"(?:daily\s+at|every\s+day\s+at)\s+(\d{1,2})(?::([0-5]\d))?\s*(am|pm)\b"
        r"|(\d{1,2})(?::([0-5]\d))?\s*(am|pm)\s+(?:daily|every\s+day)\b",
        text,
    )
    if daily_meridiem:
        hour = daily_meridiem.group(1) or daily_meridiem.group(4)
        minute = daily_meridiem.group(2) or daily_meridiem.group(5)
        period = daily_meridiem.group(3) or daily_meridiem.group(6)
        return _daily(int(hour), int(minute or 0), period)

    daily = re.search(r"(?:daily\s+at|daily|every\s+day\s+at)\s+(\d{1,2}):(\d{2})", text)
    if daily:
        return _daily(int(daily.group(1)), int(daily.group(2)), None)

    # "매일 09:00" and "매일 아침 09:00" — a spoken period in front of a clock
    # time is how the Configurator itself writes schedules back to the user.
    korean_daily_clock = re.search(
        r"매일\s*(오전|오후|아침|저녁|밤|새벽)?\s*(\d{1,2}):([0-5]\d)",
        text,
    )
    if korean_daily_clock:
        return _daily(
            int(korean_daily_clock.group(2)),
            int(korean_daily_clock.group(3)),
            korean_daily_clock.group(1),
        )

    # "매일 아침 9시", "매일 오후 3시 30분" — the way people actually say it.
    korean_daily_hour = re.search(
        r"매일\s*(오전|오후|아침|저녁|밤|새벽)?\s*(\d{1,2})\s*시"
        r"(?:\s*(\d{1,2})\s*분)?",
        text,
    )
    if korean_daily_hour:
        return _daily(
            int(korean_daily_hour.group(2)),
            int(korean_daily_hour.group(3) or 0),
            korean_daily_hour.group(1),
        )

    return None


_SCHEDULE_FRAGMENT_SPLIT_RE = re.compile(r"[\n\r.!?;,]+")

# What `agent/schedule_store.py::_validate_expression` will actually accept.
# Quoted back to the user when their phrasing is outside it, because "could not
# schedule that" without saying what *is* schedulable leaves them guessing.
SUPPORTED_SCHEDULE_FORMS = (
    "N분/시간/일 간격 (최소 60초)",
    "매일 HH:MM",
)


_SCHEDULE_FRAGMENT_LEAD_RE = re.compile(r"^(?:그리고|그럼|그러면|또|또한|and)\s+", re.IGNORECASE)


def _clean_schedule_fragment(fragment: str) -> str:
    """Strip the noise around a quoted schedule phrase, keeping the words.

    The phrase is shown back to the user and stored as ``requested_text``, and
    it is lifted out of chat prose: markdown emphasis and the conjunction the
    sentence started with ("그리고 **매일 오전 9시에**") are artefacts of where
    it was quoted from, not of what the user asked for.
    """
    cleaned = fragment.replace("**", "").replace("__", "").strip()
    cleaned = _SCHEDULE_FRAGMENT_LEAD_RE.sub("", cleaned).strip()
    return cleaned or fragment.strip()


def _first_schedule_fragment(text: str) -> str | None:
    """Return the smallest piece of ``text`` that reads as a schedule.

    Splitting first keeps the quoted phrase short enough to show back to the
    user ("매일 아침 9시에" rather than the whole paragraph it sat in), and it
    keeps the parser's own "no schedule" words (수동, manual, …) scoped to the
    clause they appear in.
    """
    if not isinstance(text, str):
        return None
    for fragment in _SCHEDULE_FRAGMENT_SPLIT_RE.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue
        if _schedule_expression_from_draft(fragment) is not None:
            return _clean_schedule_fragment(fragment)[:120]
    return None


def _schedule_display_name(expression: dict[str, Any]) -> str:
    """Name a schedule after what it does, not after the sentence it came from.

    ``name=phrase`` stored whatever chat fragment happened to parse — "그리고
    **매일 오전 9시에** flutter test 돌려서 실패하면 알려줘" was a real schedule
    name — and that string is the label every list, every edit dialog and every
    notification shows. This is a deterministic rendering of the *parsed*
    expression, so it can only say something the schedule actually does; the
    user's own wording is kept verbatim in ``requested_text``.
    """
    kind = str(expression.get("kind") or "")
    if kind == "daily_at":
        return f"매일 {expression.get('time')}"
    if kind == "interval":
        seconds = int(expression.get("seconds") or 0)
        if seconds % 86400 == 0 and seconds >= 86400:
            return f"{seconds // 86400}일마다"
        if seconds % 3600 == 0 and seconds >= 3600:
            return f"{seconds // 3600}시간마다"
        return f"{max(seconds // 60, 1)}분마다"
    return "예약 실행"


def _conversation_schedule_phrase(session: BuilderSession) -> str | None:
    """Read the conversation's last word on when this agent should run.

    Newest first, and the first message that expresses *any* timing intent
    settles it: a user who says "매일 9시" and then "아니야, 수동으로" gets no
    schedule, while silence after a stated time leaves that time standing.

    Assistant turns count as a statement of the schedule because that is how
    the defect actually reached the user: they described the job in prose and
    the Configurator was the one that said "매일 아침 09:00 자동 실행". Only the
    user can revoke it, though — the assistant musing about manual runs is not
    a decision.
    """
    for message in reversed(session.messages):
        role = message.get("role")
        content = message.get("content")
        if role == "system" or not isinstance(content, str) or not content.strip():
            continue
        phrase = _first_schedule_fragment(content)
        if phrase is not None:
            return phrase
        if role == "user" and looks_like_manual_timing(content):
            return None
    return None


def _task_draft_for_commit(
    session: BuilderSession,
    draft: AgentDraft,
    task_draft: TaskDraft | None,
) -> tuple[TaskDraft | None, str | None]:
    """Reconcile the committed task draft with what the conversation promised.

    A conversation that establishes a recurring run and a commit that carries
    no ``task_draft`` is a contradiction, and until now the commit won it
    silently: no task, therefore no schedule, therefore an agent that could
    never fire while the user had just been told it runs every morning.

    It is resolved *here*, by synthesising the task, rather than by demanding
    the Configurator always emit a ``task_draft`` block. The prompt already
    asks for one; the model still omits it, and no prompt wording makes that
    guaranteed. Nothing is invented in the process — the goal is the agent's
    own description (the thing the user just approved) and the schedule is the
    literal phrase they or the Configurator wrote. The alternative, refusing
    the commit, would throw away a finished agent over a missing echo of what
    the conversation already said.

    Returns the draft plus the origin recorded in the commit result, so the
    caller can say where the task it created came from.
    """
    if task_draft is not None and (task_draft.schedule or "").strip():
        return task_draft, "task_draft"

    phrase = _conversation_schedule_phrase(session)
    if phrase is None:
        return task_draft, ("task_draft" if task_draft is not None else None)
    if task_draft is None:
        return (
            TaskDraft(goal=task_goal_from_draft(draft), schedule=phrase),
            "synthesized_from_schedule_intent",
        )
    return (
        task_draft.model_copy(update={"schedule": phrase}),
        "task_draft_with_conversation_schedule",
    )


def _create_commit_schedule(
    *,
    task_id: str,
    task_draft: TaskDraft,
    draft: AgentDraft,
    goal: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Create the schedule the task asked for, and report either way.

    The second return value is the ``commit_result.schedule`` fact. It is never
    "created" unless a row exists, and never silent when one does not: the
    previous ``except ValueError: pass`` is the reason an unschedulable phrase
    produced an agent that looked scheduled.
    """
    phrase = (task_draft.schedule or "").strip()
    if not phrase:
        return None, {
            "created": False,
            "reason": "no_schedule_requested",
            "message": "예약 시각을 정하지 않아 예약을 만들지 않았습니다. 이 작업은 직접 실행해야 합니다.",
        }

    expression = _schedule_expression_from_draft(phrase)
    if expression is None:
        logger.warning(
            "builder_commit_schedule_unparsed task_id=%s phrase=%r",
            task_id,
            phrase,
        )
        return None, {
            "created": False,
            "reason": "unparsed_schedule_text",
            "requested_text": phrase,
            "supported_forms": list(SUPPORTED_SCHEDULE_FORMS),
            "message": (
                f"'{phrase}'을(를) 예약 형식으로 바꾸지 못해 예약을 만들지 않았습니다. "
                f"지원 형식: {', '.join(SUPPORTED_SCHEDULE_FORMS)}."
            ),
        }

    try:
        schedule = get_schedule_store().create(
            task_id=task_id,
            expression=expression,
            name=_schedule_display_name(expression),
            provider_id=draft.provider_id,
            model=draft.model,
            cwd=task_draft.cwd,
            prompt=goal,
            enabled=True,
        )
    except ValueError as exc:
        logger.warning(
            "builder_commit_schedule_rejected task_id=%s phrase=%r error=%s",
            task_id,
            phrase,
            exc,
        )
        return None, {
            "created": False,
            "reason": "rejected_by_schedule_store",
            "requested_text": phrase,
            "expression": expression,
            "detail": str(exc),
            "message": (
                f"'{phrase}'에서 만든 예약을 저장하지 못해 예약이 없습니다: {exc}"
            ),
        }

    return schedule, {
        "created": True,
        "id": schedule["id"],
        "name": schedule["name"],
        "expression": schedule["expression"],
        "enabled": bool(schedule["enabled"]),
        "next_run_at": schedule["next_run_at"],
        "requested_text": phrase,
        "message": f"'{phrase}' 예약을 만들었습니다. 다음 실행: {schedule['next_run_at']}.",
    }


def _commit_summary(
    *,
    agent_fact: dict[str, Any],
    task_fact: dict[str, Any],
    schedule_fact: dict[str, Any],
) -> str:
    """One sentence a user can act on, covering only what actually exists."""
    lines = [f"에이전트 '{agent_fact.get('name') or agent_fact['id']}'을(를) 만들었습니다."]
    if task_fact.get("created"):
        lines.append(f"작업을 만들었습니다: {task_fact.get('goal') or task_fact['id']}.")
    else:
        lines.append("실행할 작업은 만들지 않았습니다.")
    lines.append(str(schedule_fact.get("message") or ""))
    if not schedule_fact.get("created"):
        lines.append("이 에이전트는 스스로 실행되지 않습니다. 직접 실행해야 합니다.")
    return " ".join(line for line in lines if line)


def _pseudo_agent_protected_response() -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content={"error": "pseudo_agent_protected"},
    )


def _normalize_agent_workflow(flow_json: Any) -> list[dict[str, Any]]:
    try:
        return normalize_workflow(flow_json)
    except WorkflowNormalizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _flow_input_conflict_response() -> JSONResponse:
    """The refusal for a body that carries both ``flow_json`` and ``flow_graph``.

    Such a request is ambiguous about which representation is authoritative,
    and picking one silently would mean the other — something the caller
    explicitly wrote — is discarded without a word. The refusal names both
    fields so the fix is obvious: send exactly one.
    """
    detail = (
        "Request carries both flow_json and flow_graph. They are two"
        " representations of the same workflow, so sending both makes it"
        " ambiguous which one is meant — send exactly one of them."
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": "flow_input_conflict",
            "detail": detail,
            "message": detail,
        },
    )


def _fold_flow_graph_input(
    flow_graph: Any,
) -> tuple[list[dict[str, Any]] | None, JSONResponse | None]:
    """Fold a request's kernel-wire graph into the canonical linear flow_json.

    Returns ``(normalized_steps, None)`` on success or ``(None, refusal)``
    when the request must be refused — the same shape as
    ``_check_workflow_contract``, so the routes read the two gates alike.

    Three refusals, each explicit (agent-flow-core T-B-05):

    * **Kernel absent** — ``422`` with ``reason: kernel_not_installed``. The
      read view degrades to ``flow_graph_unavailable`` in this situation
      (see :func:`_flow_graph_view`), but a *write* cannot degrade: quietly
      dropping or half-storing a graph the user sent would lose their work.
      The imports are lazy for the same reason ``_flow_graph_view``'s are —
      the deployed venv has no ``agent_flow_core``, and a module-level import
      would stop the whole server, flow_json writes included.
    * **Not a kernel Flow** — ``400 invalid_flow_graph``: the dict does not
      validate as the wire form ``GET /agents/{id}`` serves.
    * **Outside the linear subset** — ``400 unsupported_topology`` carrying
      ``from_graph``'s *full* ``linear.*`` issue list verbatim, so the caller
      sees every violation at once, not just that folding failed.
    """
    try:
        from agent.flow_graph import UnsupportedTopologyError, from_graph
    except ImportError as exc:  # kernel absent — the deployed-venv case
        detail = (
            "This server's Python environment has no agent-flow-core kernel,"
            " so a flow_graph request body cannot be folded into flow_json."
            " Nothing was saved. Send the workflow as flow_json instead —"
            " that is the stored canon and needs no kernel."
        )
        return None, JSONResponse(
            status_code=422,
            content={
                "error": "kernel_not_installed",
                "reason": "kernel_not_installed",
                "detail": detail,
                "message": detail,
                "exception": str(exc),
            },
        )

    # Importable iff the import above succeeded; kept lazy for the same
    # deployed-venv reason.
    from agent_flow_core.model import Flow
    from pydantic import ValidationError as _KernelValidationError

    try:
        flow = Flow.model_validate(flow_graph)
    except _KernelValidationError as exc:
        detail = f"flow_graph does not validate as a kernel Flow: {exc}"
        return None, JSONResponse(
            status_code=400,
            content={
                "error": "invalid_flow_graph",
                "detail": detail,
                "message": detail,
            },
        )

    try:
        return from_graph(flow), None
    except UnsupportedTopologyError as exc:
        return None, JSONResponse(
            status_code=400,
            content={
                "error": "unsupported_topology",
                "detail": exc.detail["message"],
                "message": exc.detail["message"],
                "issues": exc.detail["issues"],
            },
        )


class BuilderCommitBody(BuilderCommitRequest):
    """The commit request, plus the one answer only this gate ever asks for.

    ``commit_incomplete`` is not part of what a draft *is*, so it does not
    belong on ``BuilderCommitRequest``: it is the caller's reply to a refusal
    this route made ("yes, save it anyway, I know it will stall"). Keeping it
    on the route's own body model puts the escape hatch beside the gate that
    offers it and leaves the shared draft model describing only the draft.
    """

    commit_incomplete: bool = False


class AgentCreateBody(AgentCreate):
    """``AgentCreate`` with the same deliberate-incomplete escape hatch.

    ``flow_graph`` is the graph-shaped way to say what ``flow_json`` says
    (agent-flow-core T-B-05): a kernel wire-form ``Flow`` dict, exactly the
    shape ``GET /agents/{id}`` serves as ``flow_graph``. It is a route-level
    *input representation*, not a column — the route folds it into the linear
    ``flow_json`` before anything is stored, so the stored canon never
    changes shape. The two fields are mutually exclusive: a request carrying
    both is refused rather than one being silently preferred.
    """

    commit_incomplete: bool = False
    flow_graph: dict[str, Any] | None = None


class AgentUpdateBody(AgentUpdate):
    """``AgentUpdate`` with the same deliberate-incomplete escape hatch.

    ``update_agent`` builds its patch with ``exclude_unset=True`` and hands it
    to the store, so this field must be popped there before it reaches a column
    that does not exist.

    ``flow_graph`` (see :class:`AgentCreateBody`) is popped the same way: it
    is folded into a ``flow_json`` patch entry before the patch reaches the
    store, and never travels as its own column.
    """

    commit_incomplete: bool = False
    flow_graph: dict[str, Any] | None = None


def _flow_has_browser_step(flow: Any) -> bool:
    if not isinstance(flow, list):
        return False
    return any(
        isinstance(step, dict)
        and str(step.get("type") or step.get("step_type") or "").strip().lower()
        in BROWSER_STEP_TYPES
        for step in flow
    )


async def _browser_readiness_for_contract(flow: Any) -> dict[str, Any] | None:
    """The readiness snapshot the contract check should judge against.

    Cache first (``get_cached_browser_readiness_sync``): the probe starts a
    Playwright driver, and doing that on every save would make writing an agent
    pay for a diagnostic. When there is no fresh snapshot the answer is
    genuinely unknown, and unknown is never reported as ready.

    A flow with a browser step is the one case where unknown is not good
    enough — that workflow's whole fate depends on the answer — so it falls
    through to the cache-backed async getter, which probes at most once per TTL
    and then serves every other commit in that window from the cache. A flow
    with no browser step never triggers a probe at all.
    """
    cached = get_cached_browser_readiness_sync()
    if cached is not None:
        return cached
    if not _flow_has_browser_step(flow):
        return None
    try:
        return await get_browser_runtime_readiness()
    except Exception:  # pragma: no cover - diagnostics must never fail a save
        logger.warning("browser runtime readiness probe failed", exc_info=True)
        return None


async def _check_workflow_contract(
    flow: Any,
    *,
    commit_incomplete: bool,
) -> tuple[ContractReport, JSONResponse | None]:
    """Run the builder-runtime contract check for a workflow about to be saved.

    Returns the report and, when the request must be refused, the response to
    return. Refusal is a plain ``400`` — the same code every other workflow
    normalization refusal on this router answers with
    (``_normalize_agent_workflow`` above, ``workflow_v2``): one class of
    "this workflow cannot be saved as written", one status code. A second code
    for the same class would only teach clients to branch on which validator
    happened to speak first.

    ``commit_incomplete=True`` saves anyway. That is not a bypass of the
    judgement — the findings are still reported back in
    ``commit_result.readiness`` — it is the answer to a real case: a draft
    someone wants to keep working on tomorrow, whose browser target they do not
    know yet. The runtime adapter still parks such a run honestly.
    """
    readiness = await _browser_readiness_for_contract(flow)
    report = analyze_workflow(flow, browser_readiness=readiness)
    if report.has_blocking and not commit_incomplete:
        return report, _contract_refusal_response(report)
    return report, None


def _contract_refusal_response(report: ContractReport) -> JSONResponse:
    """The refusal a client can act on, item by item.

    ``detail`` is a sentence because every existing client renders that field
    verbatim (``lib/services/builder_service.dart`` reads
    ``detail ?? error ?? message``), so an app that knows nothing about this
    check still tells the user something true. ``unresolved`` /
    ``unknown_step_references`` carry the same facts in a shape a builder UI
    can turn into one input box per unresolved target — ``step_id`` and a
    **0-based** ``action_index`` address the exact action to patch.
    """
    unresolved = [
        finding.to_dict()
        for finding in report.by_code(CODE_UNRESOLVED_BROWSER_TARGET)
    ]
    unknown_refs = [
        finding.to_dict() for finding in report.by_code(CODE_UNKNOWN_STEP_REFERENCE)
    ]
    error = (
        "unresolved_browser_targets"
        if unresolved
        else ("unknown_step_reference" if unknown_refs else "workflow_contract_blocked")
    )
    asks = [finding.ask for finding in report.blocking if finding.ask]
    detail = " ".join(
        [
            "저장하지 않았습니다. 지금 저장하면 실행할 때 반드시 멈추는 단계가 "
            f"{len(report.blocking)}개 있습니다:",
            *asks,
            "값을 채운 뒤 다시 저장하거나, 미완성 상태로 남겨 두려면 "
            "commit_incomplete=true 로 요청하세요.",
        ]
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": error,
            "detail": detail,
            "message": detail,
            "unresolved": unresolved,
            "unknown_step_references": unknown_refs,
            "blocking": [finding.to_dict() for finding in report.blocking],
            "warnings": [finding.to_dict() for finding in report.warnings],
            "can_save_incomplete": True,
        },
    )


def _contract_readiness_fact(
    report: ContractReport,
    *,
    saved_incomplete: bool,
) -> dict[str, Any]:
    """What the saved workflow still cannot do, stated on the success response.

    A commit that returns 200 with a missing browser runtime is not a clean
    commit — it is a saved workflow that will park at its first browser step
    until someone installs Chromium on the server. The install is not the
    author's to perform, so refusing their work would hold it hostage to
    someone else's machine; saying nothing would let them believe it runs.
    This says it.
    """
    runtime_findings = report.by_code(CODE_BROWSER_RUNTIME_UNAVAILABLE)
    unresolved = [
        finding.to_dict()
        for finding in report.by_code(CODE_UNRESOLVED_BROWSER_TARGET)
    ]
    unknown_refs = [
        finding.to_dict() for finding in report.by_code(CODE_UNKNOWN_STEP_REFERENCE)
    ]
    warnings = [finding.to_dict() for finding in report.warnings]

    browser_runtime: dict[str, Any] | None = None
    if runtime_findings:
        finding = runtime_findings[0]
        # The command is carried only when a download is genuinely the fix.
        # The phone renders a copyable install command whenever it is present,
        # and a server that is unready because of its *browser setting* — say,
        # set to installed Chrome on a machine that has none — would otherwise
        # be answered with an install that changes nothing.
        needs_download = finding.detail.get("install_required") is not False
        browser_runtime = {
            "ready": False,
            "install_command": (
                finding.detail.get("install_command") if needs_download else None
            ),
            "install_required": needs_download,
            "message": finding.ask,
            "step_ids": finding.detail.get("step_ids") or [],
        }

    messages = [finding.ask for finding in report.findings if finding.ask]
    # `ok` answers one question — "will this workflow run as written?" — so it
    # is false for a blocking finding and for a missing browser runtime, and
    # true despite an advisory `possible_unknown_step_reference`, which is a
    # note about prose and stops nothing. The full list is still in `warnings`.
    return {
        "ok": not report.has_blocking and not runtime_findings,
        "browser_runtime": browser_runtime,
        "unresolved_targets": unresolved,
        "unknown_step_references": unknown_refs,
        "warnings": warnings,
        "saved_incomplete": bool(saved_incomplete),
        "message": " ".join(messages),
    }


async def run_configurator_turn(
    session: BuilderSession,
    *,
    timeout: float = 120.0,
) -> str:
    """Run one Configurator turn through the selected LLM provider."""

    selection = get_chat_provider_selection()
    llm_session = await create_chat_session(
        f"agent-builder-{session.session_id}",
        str(Path.cwd()),
        selection,
    )
    prompt = build_configurator_turn_prompt(session)
    try:
        return await asyncio.wait_for(
            _collect_llm_response_text(llm_session, prompt),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        with suppress(Exception):
            await llm_session.abort_current_turn()
        raise


async def _collect_llm_response_text(llm_session: Any, prompt: str) -> str:
    chunks: list[str] = []
    result_text: str | None = None
    async for event in llm_session.send_message(prompt):
        event_type = event.get("type")
        if event_type == "result":
            result = event.get("result")
            if isinstance(result, str):
                result_text = result
            break
        if event_type == "assistant":
            text = _extract_assistant_text(event.get("message"))
            if text:
                chunks.append(text)
            continue
        if event_type == "error":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            raise RuntimeError(message or "LLM provider returned an error")
    return result_text if result_text is not None else "".join(chunks)


def _extract_assistant_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _validate_commit_draft(draft: AgentDraft) -> None:
    missing: list[str] = []
    if not (draft.name or "").strip():
        missing.append("name")
    if not draft.system_prompt.strip():
        missing.append("system_prompt")
    if not draft.provider_id:
        missing.append("provider_id")
    if not draft.flow:
        missing.append("flow")
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"AgentDraft missing required field(s): {', '.join(missing)}",
        )


def _reject_unapproved_shell_steps(draft: AgentDraft) -> None:
    """Refuse a shell step that names no script.

    A ``shell`` step runs the script its ``script_id`` names. With that field
    empty the step names nothing, and the agent was committed referring to work
    that exists in no registry — the reported session saved a `run_flutter_test`
    step whose output the next step then "summarised". The step id is in the
    message because that is what the user sees in the flow editor, and the
    pending proposal's name is there because approving it is the fix.

    ``normalize_workflow`` refuses the same thing (``agent/workflow_v2.py``),
    with a message written for whoever wrote the JSON. This runs first so the
    person who never saw any JSON gets told which of *their* steps is unbuilt
    and what to approve. Both refusals answer with 400 — see RESULT_004.
    """
    pending_by_step = {
        (request.step_id or "").strip(): request.name.strip()
        for request in draft.script_requests
        if (request.step_id or "").strip() and request.name.strip()
    }
    unassigned = [
        request.name.strip()
        for request in draft.script_requests
        if not (request.step_id or "").strip() and request.name.strip()
    ]
    offenders: list[str] = []
    for index, step in enumerate(draft.flow, start=1):
        if step.type != "shell":
            continue
        if str(getattr(step, "script_id", "") or "").strip():
            continue
        step_id = (step.id or "").strip() or f"step_{index}"
        proposal = pending_by_step.get(step_id) or (unassigned[0] if unassigned else None)
        if proposal:
            offenders.append(
                f"'{step_id}' (승인 대기 중인 스크립트 제안: '{proposal}')"
            )
        else:
            offenders.append(f"'{step_id}' (연결된 스크립트 제안이 없습니다)")
    if offenders:
        raise HTTPException(
            status_code=400,
            detail=(
                "셸 스텝에 script_id가 없어 저장하지 않았습니다: "
                + ", ".join(offenders)
                + ". 스크립트를 승인하면 해당 단계에 연결됩니다."
            ),
        )


def _resolve_commit_task_draft(
    body: BuilderCommitRequest,
    session: BuilderSession,
) -> TaskDraft | None:
    body_payload = body.model_dump()
    legacy_task_request = body_payload.get("also_create_task")
    if body.task_draft is not None and _task_draft_has_content(body.task_draft):
        return body.task_draft
    if isinstance(legacy_task_request, TaskDraft) and _task_draft_has_content(
        legacy_task_request
    ):
        return legacy_task_request
    if isinstance(legacy_task_request, dict):
        legacy_task_draft = TaskDraft.model_validate(legacy_task_request)
        if _task_draft_has_content(legacy_task_draft):
            return legacy_task_draft
    if session.task_draft is not None and _task_draft_has_content(session.task_draft):
        return session.task_draft
    return None


def _task_draft_has_content(task_draft: TaskDraft) -> bool:
    return any(
        isinstance(value, str) and value.strip()
        for value in (
            task_draft.goal,
            task_draft.schedule,
            task_draft.cwd,
            task_draft.workspace_id,
        )
    )


def _get_or_create_builder_session(body: BuilderTurn) -> BuilderSession:
    session = get_builder_session(body.session_id)
    if session is None:
        system_prompt = build_configurator_system_prompt([])
        session = create_builder_session(
            system_prompt=system_prompt,
            draft=body.draft,
        )
    else:
        session.set_client_draft(body.draft)
    return session


def _builder_response(
    session: BuilderSession,
    *,
    assistant_message: str,
    status: str = "completed",
    job_id: str | None = None,
    error: str | None = None,
) -> BuilderTurnResponse:
    return BuilderTurnResponse(
        session_id=session.session_id,
        assistant_message=assistant_message,
        updated_draft=session.current_draft,
        is_ready_to_commit=session.is_ready_to_commit,
        should_offer_task=False,
        task_draft=session.task_draft,
        # Every response carries the conversation's script proposals, not just
        # the turn that raised one: the body arrives after the turn is answered
        # (see routes.script_proposals), so a client that only ever reads turns
        # still catches up on the next one.
        script_proposals=[
            proposal_view(proposal)
            for proposal in list_proposals_for_session(session.session_id)
        ],
        status=status,
        job_id=job_id,
        # Kept in the wire format for older clients, but nothing produces a
        # fallback draft any more — a failed LLM call reports status=failed.
        fallback=False,
        error=error,
    )


def _configurator_failure_response(
    session: BuilderSession,
    *,
    reason: str | None = None,
    job_id: str | None = None,
) -> BuilderTurnResponse:
    """Report an LLM failure as a failure.

    This used to synthesise a draft with server-side rules whenever the
    Configurator LLM timed out or errored. That draft looked like a real
    answer, so a transport failure read as "the model produced nonsense" —
    worse than returning nothing. The draft is left untouched and the caller
    sees status=failed plus the real reason.
    """
    session.touch()
    return _builder_response(
        session,
        assistant_message=_configurator_failure_message(reason=reason),
        status="failed",
        job_id=job_id,
        error=reason,
    )


def _configurator_failure_message(*, reason: str | None = None) -> str:
    parts = [
        "에이전트 빌더 LLM 호출이 실패했습니다. 초안은 만들지 않았습니다.",
        "잠시 후 다시 시도하거나, 서버의 LLM 설정을 확인해 주세요.",
    ]
    if reason:
        parts.append(f"원인: {reason}")
    return "\n".join(parts)


def _apply_successful_configurator_response(
    session: BuilderSession,
    *,
    raw_response: str,
    user_message: str,
    job_id: str | None = None,
) -> BuilderTurnResponse:
    parsed = session.apply_llm_response(
        raw_response,
        user_message=user_message,
    )
    # A script the Configurator asked for becomes a proposal here, in
    # `drafting`. Nothing is written or registered by this call — it only
    # records what was asked for, so the drafting job (kicked off once the
    # turn has been answered) knows what to write.
    sync_proposals_for_session(
        session.session_id,
        session.current_draft.script_requests,
    )
    session.touch()
    return _builder_response(
        session,
        assistant_message=parsed.assistant_message,
        status="completed",
        job_id=job_id,
    )


def _clear_expired_builder_converse_jobs() -> None:
    now = datetime.now(UTC)
    expired = [
        job_id
        for job_id, job in BUILDER_CONVERSE_JOBS.items()
        if now - job.updated_at > BUILDER_CONVERSE_JOB_TTL
    ]
    for job_id in expired:
        BUILDER_CONVERSE_JOBS.pop(job_id, None)


def _job_payload(job: BuilderConverseJob) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job.id,
        "session_id": job.session_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }
    if job.error:
        payload["error"] = job.error
    if job.response is not None:
        payload["result"] = job.response.model_dump(mode="json", exclude_none=True)
        # Duplicate the completed result's legacy fields at the top level so a
        # thin client can parse the final poll response like BuilderTurnResponse.
        payload.update(job.response.model_dump(mode="json", exclude_none=True))
        payload["status"] = job.status
    return payload


async def _run_builder_converse_job(job_id: str) -> None:
    job = BUILDER_CONVERSE_JOBS.get(job_id)
    if job is None:
        return
    session = get_builder_session(job.session_id)
    if session is None:
        job.status = "failed"
        job.error = "Builder session expired"
        job.touch()
        return
    if not session.lock.acquire(blocking=False):
        job.status = "failed"
        job.error = "Builder session is already processing a request"
        job.touch()
        return
    try:
        job.status = "running"
        job.touch()
        raw_response = await run_configurator_turn(session, timeout=120.0)
        job.response = _apply_successful_configurator_response(
            session,
            raw_response=raw_response,
            user_message=job.user_message,
            job_id=job.id,
        )
        job.status = "completed"
        job.error = None
    except asyncio.TimeoutError:
        job.error = "Configurator LLM timed out."
        job.response = _configurator_failure_response(
            session,
            reason=job.error,
            job_id=job.id,
        )
        job.status = "failed"
    except RuntimeError as exc:
        job.error = str(exc)
        job.response = _configurator_failure_response(
            session,
            reason=job.error,
            job_id=job.id,
        )
        job.status = "failed"
    finally:
        job.touch()
        session.lock.release()

    # Outside the lock, and after the job has settled: the client is already
    # reading the reply while the script gets written. A drafting failure is
    # the proposal's failure, not the turn's, so it must not touch job.status.
    try:
        await draft_pending_proposals(job.session_id)
    except Exception as exc:  # noqa: BLE001 - never let this fail a finished turn
        logger.warning("builder_script_proposal_drafting_failed error=%s", exc)


def _validate_task_draft(task_draft: TaskDraft) -> None:
    goal = getattr(task_draft, "goal", None)
    if not isinstance(goal, str) or not goal.strip():
        raise HTTPException(
            status_code=422,
            detail="TaskDraft missing required field: goal",
        )


@router.post(
    "/builder/converse",
    dependencies=[Depends(verify_api_key)],
    response_model=BuilderTurnResponse,
    response_model_exclude_none=True,
)
async def builder_converse(
    body: BuilderTurn,
    background_tasks: BackgroundTasks,
) -> BuilderTurnResponse:
    session = _get_or_create_builder_session(body)

    if not session.lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Builder session is already processing a request",
        )

    try:
        session.touch()
        session.append_user_message(body.user_message)
        try:
            raw_response = await run_configurator_turn(
                session,
                timeout=BUILDER_CONVERSE_FAST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            return _configurator_failure_response(
                session,
                reason="Configurator LLM timed out before the fast response window.",
            )
        except RuntimeError as exc:
            return _configurator_failure_response(session, reason=str(exc))

        response = _apply_successful_configurator_response(
            session,
            raw_response=raw_response,
            user_message=body.user_message,
        )
        # Drafting runs after this response is on the wire. It is a second LLM
        # call with its own 180s budget, and this route answers inside 20s or
        # tells the client to poll — waiting for it here would make every turn
        # that asks for a script look like a hung builder.
        background_tasks.add_task(draft_pending_proposals, session.session_id)
        return response
    finally:
        session.lock.release()


@router.post(
    "/builder/converse/jobs",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def create_builder_converse_job(
    body: BuilderTurn,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _clear_expired_builder_converse_jobs()
    session = _get_or_create_builder_session(body)
    if session.lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Builder session is already processing a request",
        )

    session.touch()
    session.append_user_message(body.user_message)
    job_id = f"builder_job_{uuid.uuid4().hex}"
    # Seed the job with a plain "queued" acknowledgement. It used to be seeded
    # with a rule-built draft so a poll always had something to show; that made
    # a pending — or failing — LLM call look like it had already answered.
    queued = _builder_response(
        session,
        assistant_message="에이전트 빌더가 응답을 생성하고 있습니다.",
        status="queued",
        job_id=job_id,
    )
    job = BuilderConverseJob(
        id=job_id,
        session_id=session.session_id,
        user_message=body.user_message,
        status="queued",
        response=queued,
    )
    BUILDER_CONVERSE_JOBS[job_id] = job
    background_tasks.add_task(_run_builder_converse_job, job_id)
    return _job_payload(job)


@router.get(
    "/builder/converse/jobs/{job_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def get_builder_converse_job(job_id: str) -> dict[str, Any]:
    _clear_expired_builder_converse_jobs()
    job = BUILDER_CONVERSE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Builder job '{job_id}' not found")
    return _job_payload(job)


@router.post(
    "/builder/commit",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def builder_commit(body: BuilderCommitBody) -> dict[str, Any] | JSONResponse:
    """Commit the draft, and report exactly what now exists.

    ``commit_result`` is the point of this endpoint's response, not a decoration
    on it: agent, task and schedule each either created (with its id) or not
    (with a machine-readable ``reason`` and a sentence naming what could not be
    built). Success used to be indistinguishable from "created a name and
    skipped the rest", which is how a user came away believing an agent with no
    task and no schedule ran every morning.
    """
    session = get_builder_session(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Builder session '{body.session_id}' not found",
        )

    draft = body.draft
    task_draft = _resolve_commit_task_draft(body, session)
    # Commit saves the draft the user read and approved. It deliberately does
    # not re-run `enrich_draft_from_user_intent` here: that pass back-filled
    # missing required fields (which `_validate_commit_draft` must reject with
    # 422 instead of quietly inventing), re-extracted a schedule that
    # `_task_draft_for_commit` already derives from the conversation, and —
    # reading only the draft's own text — could replace an approved workflow
    # with a keyword-matched template between the screen and the database.
    task_draft, task_origin = _task_draft_for_commit(session, draft, task_draft)
    _validate_commit_draft(draft)
    _reject_unapproved_shell_steps(draft)
    if task_draft is not None:
        _validate_task_draft(task_draft)

    # The contract check runs before anything is written: a workflow that is
    # guaranteed to park at runtime must not first become an agent, a task and
    # a schedule that someone then has to clean up. The builder session is left
    # intact on refusal so the author can fix the draft and commit again.
    flow_json = _normalize_agent_workflow([step.model_dump() for step in draft.flow])
    contract_report, refusal = await _check_workflow_contract(
        flow_json,
        commit_incomplete=body.commit_incomplete,
    )
    if refusal is not None:
        return refusal

    store = _store()
    agent = store.create_agent(
        name=(draft.name or "").strip(),
        description=draft.description,
        system_prompt=draft.system_prompt,
        provider_id=draft.provider_id,
        model=draft.model,
        tools_json=[tool.model_dump() for tool in draft.tools],
        flow_json=flow_json,
        policy_overrides_json={},
    )
    for memory_seed in draft.memory_seeds:
        if memory_seed.strip():
            store.add_memory(
                agent_id=agent["id"],
                content=memory_seed.strip(),
                source_event_type="builder_seed",
            )

    result: dict[str, Any] = {"agent": agent}
    agent_fact: dict[str, Any] = {
        "created": True,
        "id": agent["id"],
        "name": agent.get("name"),
    }
    if task_draft is not None:
        goal = str(task_draft.goal).strip()
        # Resolve the folder the conversation named *here*, at the moment the
        # task is written, so a `cwd` that matched nothing never reaches the
        # metadata `_resolve_project_path` reads. `project_name` comes back set
        # only when the value matched a project the user actually registered —
        # and it has to be passed on, because every other consumer keys off it:
        # a task without one runs under the `__global__` sentinel, which puts
        # its files outside any workspace (so the path guard asks permission for
        # routine reads) and leaves a standing rule with no project to attach
        # to. `POST /tasks` has always set it; the builder path never did, which
        # is why conversationally-built agents were the ones that stalled.
        workdir = resolve_task_draft_workdir(task_draft)
        task_draft = workdir.task_draft or task_draft
        task = store.create_task(
            title=goal[:80],
            description=None,
            project_name=workdir.project_name,
            workspace_id=task_draft.workspace_id,
            assigned_agent_id=agent["id"],
            kind="general",
            source="agent_builder",
            goal=goal,
            metadata={
                "builder_session_id": body.session_id,
                "schedule": task_draft.schedule,
                "cwd": task_draft.cwd,
            },
        )
        result["task"] = task
        task_fact = {
            "created": True,
            "id": task["id"],
            "goal": goal,
            "origin": task_origin or "task_draft",
        }
        schedule, schedule_fact = _create_commit_schedule(
            task_id=task["id"],
            task_draft=task_draft,
            draft=draft,
            goal=goal,
        )
        if schedule is not None:
            result["schedule"] = schedule
    else:
        task_fact = {
            "created": False,
            "reason": "no_task_intent",
            "message": "대화에서 실행할 작업이나 반복 실행 시각을 정하지 않아 작업을 만들지 않았습니다.",
        }
        schedule_fact = {
            "created": False,
            "reason": "no_task",
            "message": "예약할 작업이 없어 예약을 만들지 않았습니다.",
        }

    result["agent"] = _agent_with_next_fire(store.get_agent(agent["id"]) or agent)
    readiness_fact = _contract_readiness_fact(
        contract_report,
        saved_incomplete=bool(body.commit_incomplete and contract_report.has_blocking),
    )
    summary = _commit_summary(
        agent_fact=agent_fact,
        task_fact=task_fact,
        schedule_fact=schedule_fact,
    )
    # The summary is the sentence a user reads instead of the payload. A saved
    # workflow that will stall belongs in that sentence, not only in a field
    # nobody renders yet.
    if not readiness_fact["ok"] and readiness_fact["message"]:
        summary = f"{summary} {readiness_fact['message']}".strip()
    result["commit_result"] = {
        "agent": agent_fact,
        "task": task_fact,
        "schedule": schedule_fact,
        # The single question the user actually asked: will this run by itself?
        "runs_unattended": bool(schedule_fact.get("created") and schedule_fact.get("enabled")),
        "readiness": readiness_fact,
        "summary": summary,
    }
    delete_builder_session(body.session_id)
    return result


@router.post(
    "/builder/steps/debug",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def builder_debug_step(body: BuilderStepDebugRequest) -> dict[str, Any]:
    """Run or inspect one draft workflow step directly from Agent Builder."""

    workflow = _normalize_agent_workflow(
        [step.model_dump() for step in body.draft.flow]
    )
    if body.step_index >= len(workflow):
        raise HTTPException(status_code=422, detail="step_index out of range")

    step = workflow[body.step_index]
    step_id = str(step.get("id") or f"step_{body.step_index + 1}")
    run_id = f"builder_debug_{body.session_id or 'draft'}"
    step_type = str(step.get("type") or "llm")

    if step_type == "browser_action":
        actions = [
            action for action in step.get("actions", []) if isinstance(action, dict)
        ]
        if _builder_step_has_external_send_action(actions):
            return {
                "status": "waiting_for_user",
                "step": step,
                "message": "발송/제출 action은 빌더 디버그에서 바로 실행하지 않습니다. 승인 단계를 통과한 실제 실행에서 처리하세요.",
                "output": {
                    "wait_reason": "external_send_requires_run_approval",
                    "actions": actions,
                },
            }
        result = await execute_browser_actions(
            actions,
            context={
                "run_id": run_id,
                "step_id": step_id,
                "workflow_step_id": step_id,
                "project_name": "agent-builder",
                "project_path": str(Path.cwd()),
            },
        )
        return {
            "status": result.status,
            "step": step,
            "message": result.prompt or result.message or "Browser step debug complete.",
            "output": result.to_output(),
        }

    if step_type in {
        "manual_handoff",
        "approval_gate",
        "mcp_tool",
        "app_action",
        "android_action",
        "mobile_action",
        "device_action",
    }:
        return {
            "status": "waiting_for_user",
            "step": step,
            "message": _builder_manual_step_prompt(step),
            "output": {
                "wait_reason": step_type,
                "on_failure": step.get("on_failure"),
            },
        }

    if step_type == "condition":
        return {
            "status": "completed",
            "step": step,
            "message": "Condition step parsed successfully. Branch execution is not applied in Builder debug.",
            "output": {"condition": step.get("condition") or step.get("description")},
        }

    prompt = _builder_llm_step_debug_prompt(body.draft, step, body.step_index)
    if not body.execute_llm:
        return {
            "status": "completed",
            "step": step,
            "message": "LLM step prompt composed.",
            "output": {"prompt": prompt},
        }

    selection = get_chat_provider_selection()
    llm_session = await create_chat_session(
        f"agent-builder-step-debug-{body.session_id or step_id}",
        str(Path.cwd()),
        selection,
    )
    try:
        assistant_text = await asyncio.wait_for(
            _collect_llm_response_text(llm_session, prompt),
            timeout=120.0,
        )
    except asyncio.TimeoutError as exc:
        with suppress(Exception):
            await llm_session.abort_current_turn()
        raise HTTPException(
            status_code=503,
            detail="Step debug LLM timed out. Please try again.",
        ) from exc

    return {
        "status": "completed",
        "step": step,
        "message": "LLM step debug complete.",
        "output": {
            "prompt": prompt,
            "assistant_message": assistant_text,
        },
    }


def _builder_llm_step_debug_prompt(
    draft: AgentDraft,
    step: dict[str, Any],
    step_index: int,
) -> str:
    return (
        f"{draft.system_prompt.strip()}\n\n"
        "You are debugging one Agent Builder workflow step. "
        "Run only this step mentally/with your available LLM reasoning, "
        "report what you would do, what inputs are missing, and whether the "
        "step is ready for real execution.\n\n"
        f"Agent name: {draft.name or '(unnamed)'}\n"
        f"Step index: {step_index + 1}\n"
        f"Step JSON:\n{step}\n"
    ).strip()


def _builder_manual_step_prompt(step: dict[str, Any]) -> str:
    on_failure = step.get("on_failure")
    if isinstance(on_failure, dict):
        prompt = on_failure.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    description = step.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return "이 단계는 사용자 확인 또는 수동 처리가 필요합니다."


def _builder_step_has_external_send_action(actions: list[dict[str, Any]]) -> bool:
    for action in actions:
        if str(action.get("type") or "").strip().lower() != "click":
            continue
        text = " ".join(
            str(action.get(key) or "")
            for key in ("selector", "target", "text", "value", "label")
        ).casefold()
        if any(marker in text for marker in ("보내기", "send", "submit", "발송", "제출")):
            return True
    return False


@router.get(
    "/workflow/step-schema",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def get_workflow_step_schema() -> dict[str, Any]:
    """Every workflow step type's fields, ready to draw a form from.

    Single source: `agent.workflow_step_schema` derives this from the same
    `WORKFLOW_STEP_SCHEMA` that `normalize_workflow_step` enforces, so a
    field a client renders here is guaranteed to be one the server will
    accept, and a field the server accepts cannot silently go unrendered —
    see AGENT_COMPOSITION_SPEC.md Phase 2.
    """
    return get_workflow_step_schema_payload()


@router.get("/browser-runtime/readiness", dependencies=[Depends(verify_api_key)], response_model=None)
async def browser_runtime_readiness() -> dict[str, Any]:
    """Return Playwright/Chromium readiness diagnostics for browser workflows."""
    return await get_browser_runtime_readiness()


@router.get("/agents", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_agents(
    include_archived: bool = False,
    include_pseudo: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    store = _store()
    agents = store.list_agents(
        include_archived=include_archived,
        include_pseudo=include_pseudo,
        limit=limit,
    )
    total = store.count_agents(
        include_archived=include_archived,
        include_pseudo=include_pseudo,
    )
    return {"agents": [_agent_with_next_fire(agent) for agent in agents], "total": total}


@router.post("/agents", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_agent(body: AgentCreateBody) -> dict[str, Any] | JSONResponse:
    """Create an agent — unless its workflow cannot run as written.

    The same gate as ``builder_commit``. A workflow does not become safe by
    arriving through a different door: the dashboard, a script and the phone
    all reach an agent definition through this route, so a check that only the
    builder performed would be a check the product does not have.
    """
    if body.id in PSEUDO_AGENT_IDS:
        return JSONResponse(
            status_code=409,
            content={"error": "agent_id_conflict"},
        )
    if body.flow_graph is not None and body.flow_json is not None:
        return _flow_input_conflict_response()
    if body.flow_graph is not None:
        # Graph input is folded to the linear canon *before* the contract
        # gate below, so a workflow arriving as a graph faces exactly the
        # same judgement as one arriving as flow_json — a different request
        # shape is not a different door.
        flow_json, fold_refusal = _fold_flow_graph_input(body.flow_graph)
        if fold_refusal is not None:
            return fold_refusal
    else:
        flow_json = _normalize_agent_workflow(body.flow_json)
    contract_report, refusal = await _check_workflow_contract(
        flow_json,
        commit_incomplete=body.commit_incomplete,
    )
    if refusal is not None:
        return refusal
    agent = _store().create_agent(
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        provider_id=body.provider_id,
        model=body.model,
        tools_json=body.tools_json,
        flow_json=flow_json,
        policy_overrides_json=body.policy_overrides_json,
    )
    payload = _agent_with_next_fire(agent)
    readiness_fact = _contract_readiness_fact(
        contract_report,
        saved_incomplete=bool(body.commit_incomplete and contract_report.has_blocking),
    )
    # Attached only when there is something to say. An agent payload is the
    # same shape everywhere it is read; a permanent `readiness: {"ok": true}`
    # on this one route would suggest `GET /agents` should carry it too.
    if not readiness_fact["ok"]:
        payload["readiness"] = readiness_fact
    return payload


@router.get("/agents/{agent_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_agent(agent_id: str) -> dict[str, Any]:
    """One agent, including the derived graph view of its workflow.

    This is the only agent read that carries ``flow_graph`` — see
    ``_agent_with_next_fire`` for why the list route does not. When the graph
    cannot be derived (kernel absent on this server, or a stored workflow that
    does not fold), ``flow_graph_unavailable`` takes its place and says which;
    the response stays 200 either way, because everything else about the agent
    is still true.
    """
    return _agent_with_next_fire(_require_agent(agent_id), include_flow_graph=True)


@router.post("/agents/{agent_id}/dry-run", dependencies=[Depends(verify_api_key)], response_model=None)
async def start_dry_run(
    agent_id: str,
    body: DryRunRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    agent = _require_agent(agent_id)
    task = _resolve_agent_task(agent_id, body.task_id)
    goal = (task.get("goal") or task.get("title")) if task else "Dry run preview"
    run = _store().create_run(
        agent_id=agent_id,
        task_id=task.get("id") if task else None,
        provider_id=agent.get("provider_id"),
        model=agent.get("model"),
        title=f"Dry run: {agent.get('name') or agent_id}",
        goal=str(goal or "Dry run preview"),
        cwd=_task_cwd(task),
        dry_run=True,
    )
    background_tasks.add_task(
        execute_task_orchestration,
        {
            "dry_run": True,
            "run_id": run["id"],
            "task_id": task.get("id") if task else None,
            "agent_id": agent_id,
            "agent": agent,
            "task": task,
        },
    )
    return {"run_id": run["id"], "run": run}


@router.post(
    "/agents/{agent_id}/run-once",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def run_agent_once(
    agent_id: str,
    body: AgentRunOnceRequest,
) -> dict[str, Any]:
    """Execute this agent's task for real, exactly once, right now.

    This is the fix for a real gap: before this route existed, the only
    "run it" button in the product was the dry run, and the only genuine
    execution path (``POST /tasks/{task_id}/schedules/{id}/trigger``) required
    a schedule that an agent might not have. A user who just built an agent
    from one sentence had no way to see what it would actually do — they
    could preview it forever and never once run it.

    It does not open a second execution path. It resolves the agent's task
    the same way the dry-run route does (:func:`_resolve_agent_task`) and
    then calls the exact same ``prepare_task_orchestration`` /
    ``execute_task_orchestration`` pair that a scheduled fire uses
    (``agent/scheduler.py::_fire_schedule``) and that
    ``POST /tasks/{task_id}/start`` already exposes. Reusing that machinery
    instead of writing a parallel "run an agent" path is deliberate: a second
    path drifts from the scheduled one over time, and the entire point of
    this endpoint is to let someone see what will really happen at 9am — that
    guarantee only holds if it is *the same* path.

    No task, no run. A schedule fires a task, and a task is what names the
    agent — there is no direct agent-to-schedule link, so "run this agent"
    can only ever mean "run its task". Rather than fabricate an ephemeral
    task the user never created (which would run *something*, but not
    necessarily the something they meant, and would leave a task behind they
    never asked for), an agent with no assigned task gets a plain 422 saying
    it has nothing to run and why — not a 500, and not a silent no-op that
    reads as success.

    This sits on the shared, api-key-gated router rather than the
    dashboard-only one (``routes/dashboard_agents.py``), on purpose and by
    contrast with ``POST /scripts``: registering a script hands a workflow
    step the ability to point at a new executable on this machine, which is
    why that stays PC-only. Running an agent the user already approved is not
    that act — ``POST /tasks/{task_id}/start`` and
    ``POST /schedules/{id}/trigger`` are already reachable this same way from
    a paired phone, and hiding *this* route behind localhost would strand the
    one client (the app, see WAVE 3's ``lib/`` changes) that most needs to
    offer it next to the simulate button.

    If this route ever silently fell back to a preview, or the caller could
    not tell it apart from ``/dry-run`` by name alone, the defect this whole
    initiative exists to fix — dry run reading as a real run — would simply
    move here.
    """
    agent = _require_agent(agent_id)
    task = _resolve_agent_task(agent_id, body.task_id)
    if task is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "이 에이전트에는 실행할 작업이 없어 실제로 실행할 수 없습니다. "
                "예약은 작업을 실행하고 작업이 에이전트를 가리키므로, 먼저 이 "
                "에이전트에 작업을 만들어야 합니다."
            ),
        )

    try:
        prepared = prepare_task_orchestration(
            task["id"],
            provider_id=body.provider_id or agent.get("provider_id"),
            model=body.model or agent.get("model"),
            cwd=body.cwd or _task_cwd(task),
            prompt=body.prompt,
            requested_capabilities=body.capabilities or None,
            auto_start=True,
            dry_run=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if prepared is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent task '{task['id']}' not found",
        )

    execution = prepared.get("execution")
    if isinstance(execution, dict):
        _spawn_background(execute_task_orchestration(execution))

    return {**prepared, "real_run": True, "dry_run": False}


@router.patch("/agents/{agent_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_agent(
    agent_id: str,
    body: AgentUpdateBody,
) -> dict[str, Any] | JSONResponse:
    """Apply a patch, minus any part of it that would be stored and never run.

    The refusal lives here, not in a screen. Every client — the phone, the
    dashboard, a script somebody wrote — reaches an agent through this route,
    so a guard that only exists in one of their UIs is a guard the product does
    not have: the next client to be written wipes the stub that explains why
    the prompt is not the prompt. See :mod:`agent.agent_origin` for which
    fields are refused and, more importantly, which are not.
    """
    patch = body.model_dump(exclude_unset=True)
    # Route-level field, never a column: strip it before the patch reaches the
    # store or the origin guard.
    commit_incomplete = bool(patch.pop("commit_incomplete", False))
    # Same for flow_graph: an input representation, not a column. Folding it
    # into a flow_json patch entry *here* means the branch below — the
    # normalize step and the contract gate — sees a graph-shaped update on
    # exactly the terms of a flow_json one.
    flow_graph = patch.pop("flow_graph", None)
    if flow_graph is not None:
        if "flow_json" in patch:
            return _flow_input_conflict_response()
        folded, fold_refusal = _fold_flow_graph_input(flow_graph)
        if fold_refusal is not None:
            return fold_refusal
        patch["flow_json"] = folded
    contract_report: ContractReport | None = None
    if "flow_json" in patch:
        patch["flow_json"] = _normalize_agent_workflow(patch["flow_json"])
        # A patch replaces the workflow wholesale, so the workflow this agent
        # will actually run is the one in the patch — checked on the same terms
        # as a fresh create.
        contract_report, refusal = await _check_workflow_contract(
            patch["flow_json"],
            commit_incomplete=commit_incomplete,
        )
        if refusal is not None:
            return refusal
    existing = _store().get_agent(agent_id)
    if existing:
        try:
            assert_patch_reaches_execution(
                agent=existing,
                patch=patch,
                origin=resolve_agent_origin(agent_id),
            )
        except AgentPromptNotEditableError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "agent_prompt_not_editable",
                    "detail": str(exc),
                    "origin": exc.origin.to_view(),
                },
            )
    try:
        agent = _store().update_agent(agent_id, patch)
    except PseudoAgentProtectedError:
        return _pseudo_agent_protected_response()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    payload = _agent_with_next_fire(agent)
    if contract_report is not None:
        readiness_fact = _contract_readiness_fact(
            contract_report,
            saved_incomplete=bool(commit_incomplete and contract_report.has_blocking),
        )
        if not readiness_fact["ok"]:
            payload["readiness"] = readiness_fact
    return payload


@router.delete("/agents/{agent_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def delete_agent(
    agent_id: str,
    archive: bool = True,
) -> dict[str, Any] | JSONResponse:
    try:
        result = _store().archive_agent(agent_id, archive=archive)
    except PseudoAgentProtectedError:
        return _pseudo_agent_protected_response()
    except AgentStoreConflictError:
        return JSONResponse(
            status_code=409,
            content={"error": "active_runs_present"},
        )
    if not result:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return result


@router.get(
    "/agents/{agent_id}/memories",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def list_agent_memories(
    agent_id: str,
    include_pinned: bool = True,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    memories = _store().list_memories(
        agent_id,
        include_pinned=include_pinned,
        limit=limit,
    )
    if memories is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {"memories": memories, "total": len(memories)}


@router.post(
    "/agents/{agent_id}/memories",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def add_agent_memory(
    agent_id: str,
    body: MemoryCreate,
) -> dict[str, Any] | JSONResponse:
    try:
        memory = _store().add_memory(
            agent_id=agent_id,
            content=body.content,
            source_run_id=body.source_run_id,
            pinned=body.pinned,
        )
    except PseudoAgentProtectedError:
        return _pseudo_agent_protected_response()
    if not memory:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return memory


@router.patch(
    "/agents/{agent_id}/memories/{memory_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def update_agent_memory(
    agent_id: str,
    memory_id: str,
    body: MemoryUpdate,
) -> dict[str, Any] | JSONResponse:
    try:
        memory = _store().update_memory(
            agent_id,
            memory_id,
            body.model_dump(exclude_unset=True),
        )
    except PseudoAgentProtectedError:
        return _pseudo_agent_protected_response()
    if not memory:
        raise HTTPException(status_code=404, detail=f"Agent memory '{memory_id}' not found")
    return memory


@router.delete(
    "/agents/{agent_id}/memories/{memory_id}",
    dependencies=[Depends(verify_api_key)],
    status_code=204,
    response_model=None,
)
async def delete_agent_memory(
    agent_id: str,
    memory_id: str,
) -> None | JSONResponse:
    try:
        deleted = _store().delete_memory(agent_id, memory_id)
    except PseudoAgentProtectedError:
        return _pseudo_agent_protected_response()
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent memory '{memory_id}' not found")
    return None


@router.get("/notifications", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_notifications(
    unread_only: bool = False,
    agent_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """What agents have left for you since you last looked."""
    from agent.notification_store import get_notification_store

    store = get_notification_store()
    return {
        "notifications": store.list_notifications(
            unread_only=unread_only, agent_id=agent_id, limit=limit
        ),
        "unread_count": store.unread_count(),
    }


@router.post(
    "/notifications/{notification_id}/read",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def mark_notification_read(notification_id: str) -> dict[str, Any]:
    from agent.notification_store import get_notification_store

    notification = get_notification_store().mark_read(notification_id)
    if notification is None:
        raise HTTPException(
            status_code=404, detail=f"Notification '{notification_id}' not found"
        )
    return {"notification": notification}


@router.post(
    "/notifications/read-all",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def mark_all_notifications_read() -> dict[str, Any]:
    from agent.notification_store import get_notification_store

    return {"marked": get_notification_store().mark_all_read()}


@router.get(
    "/agents/{agent_id}/tasks",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def list_agent_tasks(
    agent_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Work assigned to this agent, whether or not it has run yet.

    Runs are the record of execution, so a task that was created and has not
    fired leaves no trace among them. Listing what an agent was told to do —
    separately from what it did — is what makes "queued and going nowhere"
    visible instead of indistinguishable from "nothing was ever asked".
    """
    _require_agent(agent_id)
    store = _store()
    tasks = [
        task
        for task in store.list_tasks(limit=200)
        if task.get("assigned_agent_id") == agent_id
    ]
    runs_by_task: dict[str, int] = {}
    for run in store.list_runs(agent_id=agent_id, limit=200) or []:
        task_id = run.get("task_id")
        if isinstance(task_id, str):
            runs_by_task[task_id] = runs_by_task.get(task_id, 0) + 1
    for task in tasks:
        task["run_count"] = runs_by_task.get(str(task.get("id")), 0)
    return {"tasks": tasks[:limit]}


@router.get(
    "/agents/{agent_id}/schedules",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def list_agent_schedules(agent_id: str) -> dict[str, Any]:
    """Schedules that fire this agent.

    A schedule hangs off a task, and a task names its agent — three hops the
    client should not have to make to answer "when does this run".
    """
    _require_agent(agent_id)
    store = _store()
    schedule_store = get_schedule_store()
    schedules: list[dict[str, Any]] = []
    for task in store.list_tasks(limit=200):
        if task.get("assigned_agent_id") != agent_id:
            continue
        for schedule in schedule_store.list_for_task(str(task["id"])):
            schedules.append({**schedule, "task_title": task.get("title")})
    return {"schedules": schedules}


@router.post("/runs", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_run(body: AgentRunCreate) -> dict[str, Any]:
    """Create a durable agent run record."""
    run = _store().create_run(
        project_name=body.project_name,
        workspace_id=body.workspace_id,
        agent_id=body.agent_id,
        provider_id=body.provider_id,
        model=body.model,
        title=body.title,
        goal=body.goal,
        cwd=body.cwd,
        parent_run_id=body.parent_run_id,
        native_session_id=body.native_session_id,
        task_id=body.task_id,
    )
    return {"run": run}


@router.get("/runs", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_runs(
    project_name: str | None = None,
    workspace_id: str | None = None,
    agent_id: str | None = None,
    task_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """List durable agent runs."""
    return {
        "runs": _store().list_runs(
            project_name=project_name,
            workspace_id=workspace_id,
            agent_id=agent_id,
            task_id=task_id,
            status=status,
            limit=limit,
        )
    }


@router.get("/runs/{run_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_run(run_id: str) -> dict[str, Any]:
    """Return one durable agent run with messages and artifacts."""
    run = _require_run(run_id)
    store = _store()
    return {
        "run": run,
        "messages": store.list_messages(run_id),
        "artifacts": store.list_artifacts(run_id),
    }


@router.get("/runs/{run_id}/checkpoint", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_run_checkpoint(run_id: str) -> dict[str, Any]:
    """Return the active waiting checkpoint for a run."""
    checkpoint = _store().get_run_checkpoint(run_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return checkpoint


@router.post("/runs/{run_id}/resume", dependencies=[Depends(verify_api_key)], response_model=None)
async def resume_run(run_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Resume a run from its active workflow checkpoint."""
    run = _require_run(run_id)
    task_id = run.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise HTTPException(status_code=409, detail="Run is not linked to a task.")
    return await resume_task(task_id, background_tasks)


@router.post("/runs/{run_id}/feedback", dependencies=[Depends(verify_api_key)], response_model=None)
async def add_run_feedback(
    run_id: str,
    body: MemoryCreate,
) -> dict[str, Any]:
    """Convert a user's run annotation into agent memory."""
    run = _require_run(run_id)
    agent_id = run.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise HTTPException(status_code=409, detail="Agent run has no agent_id")
    memory = add_memory_from_event(
        agent_id=agent_id,
        content=body.content,
        source_run_id=run_id,
        source_event_type="user_annotation",
        pinned=body.pinned,
    )
    return {"memory": memory}


@router.post("/runs/{run_id}/message", dependencies=[Depends(verify_api_key)], response_model=None)
async def add_run_message(
    run_id: str,
    body: AgentRunMessageCreate,
) -> dict[str, Any]:
    """Append a user, assistant, system, or tool message to a run."""
    message = _store().add_message(
        run_id=run_id,
        role=body.role,
        content=body.content,
        attachments=body.attachments,
    )
    if not message:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return {"message": message}


@router.post("/runs/{run_id}/event", dependencies=[Depends(verify_api_key)], response_model=None)
async def append_run_event(
    run_id: str,
    body: AgentEventCreate,
) -> dict[str, Any]:
    """Append a normalized provider/app event to a run."""
    event = _store().append_event(
        run_id=run_id,
        event_type=body.event_type,
        provider_id=body.provider_id,
        provider_event=body.provider_event,
        app_event=body.app_event,
        call_id=body.call_id,
        parent_event_id=body.parent_event_id,
    )
    if not event:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return {"event": event}


@router.get("/runs/{run_id}/events", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_run_events(
    run_id: str,
    after_sequence: int | None = Query(default=None, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    """List events for a durable agent run."""
    _require_run(run_id)
    return {
        "events": _store().list_events(
            run_id,
            after_sequence=after_sequence,
            limit=limit,
        )
    }


@router.post("/runs/{run_id}/artifacts", dependencies=[Depends(verify_api_key)], response_model=None)
async def add_run_artifact(
    run_id: str,
    body: AgentArtifactCreate,
) -> dict[str, Any]:
    """Register an artifact for a run."""
    artifact = _store().add_artifact(
        run_id=run_id,
        kind=body.kind,
        path=body.path,
        mime_type=body.mime_type,
        metadata=body.metadata,
    )
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return {"artifact": artifact}


@router.get("/runs/{run_id}/artifacts", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_run_artifacts(run_id: str) -> dict[str, Any]:
    """List artifacts for a durable agent run."""
    _require_run(run_id)
    return {"artifacts": _store().list_artifacts(run_id)}


@router.get("/runs/{run_id}/artifacts/{artifact_id}/content", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_run_artifact_content(
    run_id: str,
    artifact_id: str,
    max_chars: int = Query(default=60000, ge=1, le=200000),
) -> dict[str, Any]:
    """Return safe inline content for generated text artifacts."""
    artifact = _require_artifact(run_id, artifact_id)
    path_value = artifact.get("path")
    if not isinstance(path_value, str) or not path_value:
        return {
            "artifact": artifact,
            "content": None,
            "truncated": False,
            "readable": False,
            "reason": "artifact has no file path",
        }

    path = Path(path_value)
    if not _is_generated_text_artifact_path(path):
        return {
            "artifact": artifact,
            "content": None,
            "truncated": False,
            "readable": False,
            "reason": "inline content is restricted to generated text artifacts",
        }
    if not path.is_file():
        return {
            "artifact": artifact,
            "content": None,
            "truncated": False,
            "readable": False,
            "reason": "artifact file not found",
        }

    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    return {
        "artifact": artifact,
        "content": text[:max_chars] if truncated else text,
        "truncated": truncated,
        "readable": True,
        "bytes": len(data),
    }


@router.get("/runs/{run_id}/artifacts/{artifact_id}/raw", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_run_artifact_raw(
    run_id: str,
    artifact_id: str,
) -> FileResponse:
    """Serve safe generated artifact files for preview/download."""
    artifact = _require_artifact(run_id, artifact_id)
    path_value = artifact.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise HTTPException(status_code=404, detail="Artifact has no file path")
    path = Path(path_value)
    if path.is_dir():
        raise HTTPException(status_code=400, detail="Artifact path is a directory")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    if not _is_safe_raw_artifact_path(path, artifact):
        raise HTTPException(status_code=403, detail="Artifact file is not available for raw access")
    media_type = artifact.get("mime_type") or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


def _is_generated_text_artifact_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(ARTIFACT_ROOT.resolve())
    except (OSError, ValueError):
        return False
    return path.suffix.lower() in {".log", ".diff", ".txt", ".json", ".md"}


def _is_safe_raw_artifact_path(path: Path, artifact: dict[str, Any]) -> bool:
    if _is_generated_text_artifact_path(path):
        return True
    if _is_safe_preview_screenshot(path, artifact):
        return True
    if artifact.get("kind") == "build_output" and _is_project_child(path, artifact):
        return True
    return False


def _is_safe_preview_screenshot(path: Path, artifact: dict[str, Any]) -> bool:
    if artifact.get("kind") not in {"device_screenshot", "browser_screenshot"}:
        return False
    if (artifact.get("mime_type") or "").lower() != "image/png":
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if artifact.get("kind") == "browser_screenshot":
        try:
            resolved.relative_to(ARTIFACT_ROOT.resolve())
        except (OSError, ValueError):
            return False
        return resolved.suffix.lower() == ".png"
    if resolved.name.startswith("code_bridge_preview_") and resolved.suffix.lower() == ".png":
        return resolved.parent in {Path("/tmp").resolve(), Path("/private/tmp").resolve()}
    return False


def _is_project_child(path: Path, artifact: dict[str, Any]) -> bool:
    metadata = artifact.get("metadata")
    project_name = metadata.get("project_name") if isinstance(metadata, dict) else None
    if not isinstance(project_name, str) or not project_name:
        return False
    project = get_project_db().get(project_name)
    root_value = project.get("path") if project else None
    if not isinstance(root_value, str) or not root_value:
        return False
    try:
        path.resolve().relative_to(Path(root_value).resolve())
    except (OSError, ValueError):
        return False
    return True


@router.post("/runs/{run_id}/abort", dependencies=[Depends(verify_api_key)], response_model=None)
async def abort_run(run_id: str) -> dict[str, Any]:
    """Mark a durable run as cancelled.

    Provider process termination will be connected in the agent runtime layer.
    """
    run = _store().update_run_status(run_id, "cancelled")
    if not run:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return {"run": run}


@router.post("/runs/{run_id}/preflight", dependencies=[Depends(verify_api_key)], response_model=None)
async def run_preflight(
    run_id: str,
    body: AgentRunPreflightCreate,
    require_approval: bool = True,
) -> dict[str, Any] | JSONResponse:
    """Run verification commands and attach a preflight summary to a run."""
    run = _require_run(run_id)
    project_name = body.project_name or run.get("project_name")
    if not project_name:
        raise HTTPException(status_code=400, detail="project_name is required for preflight")

    commands = [command.strip() for command in body.commands if command.strip()]
    if not commands:
        raise HTTPException(status_code=422, detail="At least one command is required")

    details = {"project_name": project_name, "commands": commands, "timeout": body.timeout}
    gate = evaluate_direct_action_gate(
        operation="process.terminal",
        project_name=project_name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=body.approval_id,
    )
    if not gate["allowed"]:
        return JSONResponse(
            status_code=int(gate["status_code"]),
            content=gate["payload"],
        )

    store = _store()
    store.append_event(
        run_id=run_id,
        event_type="preflight.started",
        app_event=details,
    )

    command_results: list[dict[str, Any]] = []
    for command in commands:
        result = await execute_terminal_command_for_current_server(
            project_name,
            command=command,
            timeout=body.timeout,
        )
        command_details = {"command": command, "timeout": body.timeout}
        record_api_action(
            operation="process.terminal",
            project_name=project_name,
            run_id=run_id,
            details=command_details,
            success=result.success,
            status_code=result.status_code,
        )
        record_tool_action_result(
            run_id=run_id,
            operation="process.terminal",
            project_name=project_name,
            details=command_details,
            result=result,
        )
        payload = result.as_response_fields()
        passed = (
            result.success
            and int(payload.get("exit_code", 1) or 0) == 0
            and not bool(payload.get("timed_out"))
            and not payload.get("error")
        )
        command_results.append(
            {
                "command": command,
                "passed": passed,
                "status_code": result.status_code,
                "result": payload,
            }
        )

    passed_all = all(item["passed"] for item in command_results)
    summary = {
        "project_name": project_name,
        "commands": commands,
        "passed": passed_all,
        "results": command_results,
    }
    store.append_event(
        run_id=run_id,
        event_type="preflight.completed",
        app_event=summary,
    )
    artifact = store.add_artifact(
        run_id=run_id,
        kind="agent_preflight",
        mime_type="application/json",
        metadata=summary,
    )
    record_api_action(
        operation="agent.preflight",
        project_name=project_name,
        run_id=run_id,
        details={"commands": commands, "passed": passed_all},
        success=passed_all,
        status_code=200 if passed_all else 400,
    )
    return {
        "run": _require_run(run_id),
        "passed": passed_all,
        "results": command_results,
        "artifact": artifact,
    }


@router.post("/emergency-stop", dependencies=[Depends(verify_api_key)], response_model=None)
async def emergency_stop() -> dict[str, Any]:
    """Cancel active runs and deny pending approvals.

    Provider-process termination is handled by per-session abort controls; this
    endpoint makes the durable platform state safe immediately.
    """
    cancelled_runs = _store().cancel_active_runs()
    pending = get_approval_store().list_pending()
    denied = []
    for approval in pending:
        decision = decide_approval(
            approval["id"],
            decision="deny",
            scope="once",
            reason="Emergency stop",
            approver={"type": "agent_host"},
        )
        if decision is not None:
            denied.append(approval["id"])

    record_api_action(
        operation="agent.emergency_stop",
        details={
            "cancelled_run_count": len(cancelled_runs),
            "denied_approval_count": len(denied),
        },
        success=True,
        status_code=200,
    )
    return {
        "cancelled_runs": cancelled_runs,
        "denied_approvals": denied,
        "summary": {
            "cancelled_runs": len(cancelled_runs),
            "denied_approvals": len(denied),
        },
    }


@router.post("/tasks", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_task(body: AgentTaskCreate) -> dict[str, Any]:
    """Create a tracked work task."""
    task = _store().create_task(
        title=body.title,
        description=body.description,
        project_name=body.project_name,
        workspace_id=body.workspace_id,
        run_id=body.run_id,
        assigned_agent_id=body.assigned_agent_id,
        kind=body.kind,
        source=body.source,
        goal=body.goal,
        priority=body.priority,
        due_at=body.due_at,
        labels=body.labels,
        assignee=body.assignee,
        requester=body.requester,
        acceptance=body.acceptance,
        metadata=body.metadata,
    )
    return {"task": task}


@router.get("/tasks", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_tasks(
    workspace_id: str | None = None,
    project_name: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    """List tracked work tasks."""
    return {
        "tasks": _store().list_tasks(
            workspace_id=workspace_id,
            project_name=project_name,
            kind=kind,
            status=status,
            limit=limit,
        )
    }


@router.get("/tasks/{task_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_task(task_id: str) -> dict[str, Any]:
    """Return one tracked work task."""
    task = _store().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return {"task": task}


@router.patch("/tasks/{task_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_task(task_id: str, body: AgentTaskUpdate) -> dict[str, Any]:
    """Update a tracked work task."""
    task = _store().update_task(task_id, body.model_dump(exclude_unset=True))
    if not task:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return {"task": task}


@router.post("/tasks/{task_id}/runs", dependencies=[Depends(verify_api_key)], response_model=None)
async def link_task_run(task_id: str, body: AgentTaskRunLinkCreate) -> dict[str, Any]:
    """Link an existing run to a work task."""
    link = _store().link_task_run(
        task_id=task_id,
        run_id=body.run_id,
        role=body.role,
        metadata=body.metadata,
    )
    if not link:
        raise HTTPException(status_code=404, detail="Agent task or run not found")
    task = _store().get_task(task_id)
    return {"link": link, "task": task}


@router.get("/tasks/{task_id}/timeline", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_task_timeline(task_id: str) -> dict[str, Any]:
    """Return task, linked runs, events, steps, and artifacts."""
    timeline = _store().list_task_timeline(task_id)
    if not timeline:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return timeline


@router.get("/tasks/{task_id}/checkpoint", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_task_checkpoint(task_id: str) -> dict[str, Any]:
    """Return the active waiting checkpoint for a task."""
    checkpoint = _store().get_task_checkpoint(task_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return checkpoint


@router.get("/tasks/{task_id}/browser-handoff", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_task_browser_handoff(task_id: str) -> dict[str, Any]:
    """Return the active browser handoff session for a waiting task."""
    payload = _active_browser_handoff_payload(task_id)
    if payload is None:
        if _store().get_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
        raise HTTPException(status_code=409, detail="Task has no active browser handoff.")
    return payload


@router.post("/tasks/{task_id}/browser-handoff/snapshot", dependencies=[Depends(verify_api_key)], response_model=None)
async def snapshot_task_browser_handoff(task_id: str) -> dict[str, Any]:
    """Capture a screenshot/observation for the active browser handoff session."""
    payload = _active_browser_handoff_payload(task_id)
    if payload is None:
        if _store().get_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
        raise HTTPException(status_code=409, detail="Task has no active browser handoff.")
    session = payload["browser_session"]
    actions: list[dict[str, Any]] = []
    current_url = session.get("current_url")
    if isinstance(current_url, str) and current_url:
        actions.append({"type": "navigate", "url": current_url})
    actions.append({"type": "screenshot"})
    return await _execute_browser_handoff_actions(
        task_id=task_id,
        payload=payload,
        actions=actions,
        mode="snapshot",
    )


@router.post("/tasks/{task_id}/browser-handoff/actions", dependencies=[Depends(verify_api_key)], response_model=None)
async def run_task_browser_handoff_actions(
    task_id: str,
    body: BrowserHandoffActionCreate,
) -> dict[str, Any]:
    """Apply user-guided browser actions to the active handoff session."""
    payload = _active_browser_handoff_payload(task_id)
    if payload is None:
        if _store().get_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
        raise HTTPException(status_code=409, detail="Task has no active browser handoff.")
    return await _execute_browser_handoff_actions(
        task_id=task_id,
        payload=payload,
        actions=body.actions,
        mode="actions",
    )


@router.post("/tasks/{task_id}/browser-handoff/complete", dependencies=[Depends(verify_api_key)], response_model=None)
async def complete_task_browser_handoff(
    task_id: str,
    body: AgentTaskStepRespondCreate,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Mark browser handoff complete and resume from the active checkpoint."""
    payload = _active_browser_handoff_payload(task_id)
    if payload is None:
        if _store().get_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
        raise HTTPException(status_code=409, detail="Task has no active browser handoff.")
    sensitive = _is_sensitive_browser_handoff(payload)
    session = payload["browser_session"]
    get_browser_session_store().mark_resumed(str(session["id"]))
    step = payload["step"]
    metadata = {
        **body.metadata,
        "source": body.metadata.get("source") or "browser_handoff",
        "browser_session_id": session["id"],
    }
    if sensitive:
        metadata["sensitive"] = True
    response = await _respond_to_waiting_step(
        task_id=task_id,
        step_id=str(step["id"]),
        message="Browser handoff completed." if sensitive else body.message,
        metadata=metadata,
        remember=False if sensitive else body.remember,
        resume=body.resume,
        background_tasks=background_tasks,
    )
    response["browser_session"] = get_browser_session_store().get(str(session["id"]))
    response["browser_handoff_completed"] = True
    return response


async def _execute_browser_handoff_actions(
    *,
    task_id: str,
    payload: dict[str, Any],
    actions: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    session = payload["browser_session"]
    step = payload["step"]
    run = payload.get("run") or {}
    run_id = str(session["run_id"])
    step_id = str(step["id"])
    sensitive = _is_sensitive_browser_handoff(payload)
    storage_state_path = str(Path(str(session["context_dir"])) / "storage_state.json")
    result = await execute_browser_actions(
        actions,
        context={
            "task_id": task_id,
            "run_id": run_id,
            "step_id": step_id,
            "workflow_step_id": session.get("workflow_step_id"),
            "browser_session_id": session["id"],
            "browser_context_dir": session.get("context_dir"),
            "browser_storage_state_path": storage_state_path,
            "browser_input_storage_state_path": session.get("storage_state_path"),
        },
    )
    updates: dict[str, Any] = {"status": "waiting_for_user"}
    if result.storage_state_path:
        updates["storage_state_path"] = result.storage_state_path
    observations = result.observations or []
    last_observation = observations[-1] if observations else {}
    if isinstance(last_observation, dict):
        if isinstance(last_observation.get("url"), str):
            updates["current_url"] = last_observation["url"]
        if isinstance(last_observation.get("title"), str):
            updates["title"] = last_observation["title"]
    if result.wait_reason:
        updates["handoff_reason"] = result.wait_reason
    updated_session = get_browser_session_store().update(str(session["id"]), updates)
    stored_screenshots = [] if sensitive else result.screenshots
    artifact_ids = _store_browser_handoff_artifacts(
        run_id=run_id,
        task_id=task_id,
        step_id=step_id,
        session_id=str(session["id"]),
        screenshots=stored_screenshots,
        mode=mode,
    )
    event_run_id = str(run.get("id") or run_id)
    result_output = result.to_output()
    if sensitive:
        result_output = _redact_browser_handoff_result(result_output)
    _store().append_event(
        run_id=event_run_id,
        event_type=f"task.browser_handoff.{mode}",
        app_event={
            "task_id": task_id,
            "step_id": step_id,
            "browser_session_id": session["id"],
            "result": result_output,
            "artifact_ids": artifact_ids,
            "sensitive": sensitive,
        },
    )
    return {
        "task": payload.get("task"),
        "run": payload.get("run"),
        "step": step,
        "checkpoint": payload.get("checkpoint"),
        "browser_session": updated_session,
        "result": result_output,
        "artifact_ids": artifact_ids,
    }


def _store_browser_handoff_artifacts(
    *,
    run_id: str,
    task_id: str,
    step_id: str,
    session_id: str,
    screenshots: list[str],
    mode: str,
) -> list[str]:
    artifact_ids: list[str] = []
    for index, screenshot in enumerate(screenshots, start=1):
        artifact = _store().add_artifact(
            run_id=run_id,
            kind="browser_screenshot",
            path=str(Path(screenshot).expanduser()),
            mime_type="image/png",
            metadata={
                "task_id": task_id,
                "step_id": step_id,
                "browser_session_id": session_id,
                "mode": mode,
                "index": index,
            },
        )
        artifact_id = artifact.get("id") if isinstance(artifact, dict) else None
        if isinstance(artifact_id, str):
            artifact_ids.append(artifact_id)
    return artifact_ids


def _is_sensitive_browser_handoff(payload: dict[str, Any]) -> bool:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return False
    values = [
        checkpoint.get("reason"),
        checkpoint.get("handoff_reason"),
        checkpoint.get("required_user_action"),
        checkpoint.get("prompt"),
    ]
    joined = " ".join(str(value).lower() for value in values if value is not None)
    return any(
        token in joined
        for token in (
            "login",
            "captcha",
            "2fa",
            "otp",
            "auth",
            "permission",
            "password",
            "credential",
        )
    )


def _redact_browser_handoff_result(result: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(result)
    redacted["sensitive"] = True
    redacted["screenshots"] = []
    redacted["extracted"] = []
    if "observations" in redacted:
        redacted["observations"] = [
            {
                key: value
                for key, value in observation.items()
                if key in {"url", "title", "status"}
            }
            for observation in redacted.get("observations", [])
            if isinstance(observation, dict)
        ]
    return redacted


def _active_browser_handoff_payload(task_id: str) -> dict[str, Any] | None:
    payload = _store().get_task_checkpoint(task_id)
    if payload is None or payload.get("checkpoint") is None or payload.get("step") is None:
        return None
    checkpoint = payload.get("checkpoint")
    step = payload.get("step")
    if not isinstance(checkpoint, dict) or not isinstance(step, dict):
        return None
    session_id = checkpoint.get("browser_session_id")
    if not isinstance(session_id, str) or not session_id:
        output = step.get("output")
        if isinstance(output, dict):
            fallback = output.get("browser_session_id")
            session_id = fallback if isinstance(fallback, str) else ""
    if not session_id:
        return None
    session = get_browser_session_store().get(session_id)
    if session is None or session.get("task_id") != task_id:
        return None
    if session.get("status") not in {"created", "waiting_for_user", "resumed"}:
        return None
    return {**payload, "browser_session": session}


@router.get("/tasks/{task_id}/steps", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_task_steps(task_id: str) -> dict[str, Any]:
    """Return orchestration steps for a tracked work task."""
    task = _store().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return {"task": task, "steps": _store().list_task_steps(task_id)}


@router.patch("/tasks/{task_id}/steps/{step_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_task_step(
    task_id: str,
    step_id: str,
    body: AgentTaskStepUpdate,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Update one task step after parameter review."""
    store = _store()
    step = store.get_task_step(step_id)
    if not step or step.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail=f"Agent task step '{step_id}' not found")
    payload = body.model_dump(exclude_unset=True)

    # If the caller is resolving the active checkpoint, capture the resume
    # payload BEFORE marking the step completed. `get_task_checkpoint` walks
    # the step list looking for one whose status is still in the waiting set,
    # so it would return an empty checkpoint the moment we flip this step.
    pre_resume_result: dict[str, Any] | None = None
    pre_task = None
    if payload.get("status") == "completed":
        pre_task = store.get_task(task_id)
        pre_meta = (pre_task or {}).get("metadata") or {}
        pre_cp = pre_meta.get("active_checkpoint") if isinstance(pre_meta, dict) else None
        if isinstance(pre_cp, dict) and pre_cp.get("step_id") == step_id:
            try:
                pre_resume_result = resume_task_orchestration(task_id)
            except ValueError:
                pre_resume_result = None

    updated = store.update_task_step(step_id, payload)

    # When an external caller marks the step completed and the task was
    # parked on this step's active_checkpoint, clear the checkpoint and kick
    # the workflow forward via the resume payload we captured above.
    # Otherwise the orchestrator would still see the stale checkpoint and
    # the run would stall even though the user has already resolved the gate.
    if payload.get("status") == "completed" and pre_resume_result is not None:
        metadata = dict((pre_task or {}).get("metadata") or {})
        metadata.pop("active_checkpoint", None)
        store.update_task(task_id, {"status": "running", "metadata": metadata})
        run_id = step.get("run_id") or (pre_task or {}).get("run_id")
        if isinstance(run_id, str) and run_id:
            store.update_run_status(run_id, "running")
        execution = pre_resume_result.get("execution")
        if isinstance(execution, dict):
            # Fire-and-forget on the running event loop. `_spawn_background`
            # keeps a strong reference so the task survives GC; FastAPI's
            # BackgroundTasks was unreliable here because the orchestrator
            # turn can take 60s+ and the caller frequently retry-polls the
            # same endpoint while the queue is still draining.
            _spawn_background(execute_task_orchestration(execution))
            logger.info(
                "patch_step_resume spawned task=%s step=%s run=%s",
                task_id,
                step_id,
                run_id,
            )
    return {"step": updated}


@router.post("/tasks/{task_id}/steps/{step_id}/run", dependencies=[Depends(verify_api_key)], response_model=None)
async def run_task_step(
    task_id: str,
    step_id: str,
    body: AgentTaskStepRunCreate,
) -> dict[str, Any]:
    """Run one reviewed task step through its concrete adapter."""
    result = await execute_task_step_adapter(
        task_id,
        step_id,
        input_override=body.input,
        approval_id=body.approval_id,
        require_approval=body.require_approval,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Agent task step '{step_id}' not found")
    return result


@router.post("/tasks/{task_id}/steps/{step_id}/respond", dependencies=[Depends(verify_api_key)], response_model=None)
async def respond_to_task_step(
    task_id: str,
    step_id: str,
    body: AgentTaskStepRespondCreate,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Persist a user response for a waiting workflow step."""
    return await _respond_to_waiting_step(
        task_id=task_id,
        step_id=step_id,
        message=body.message,
        metadata=body.metadata,
        remember=body.remember,
        resume=body.resume,
        background_tasks=background_tasks,
    )


async def _respond_to_waiting_step(
    *,
    task_id: str,
    step_id: str,
    message: str,
    metadata: dict[str, Any],
    remember: bool,
    resume: bool,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    try:
        result = _store().append_step_user_response(
            task_id=task_id,
            step_id=step_id,
            message=message,
            metadata=metadata,
            remember=remember,
        )
    except AgentStoreConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"Agent task step '{step_id}' not found")
    result["resume_requested"] = resume
    if resume:
        try:
            resume_result = resume_task_orchestration(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if resume_result is not None and isinstance(resume_result.get("execution"), dict):
            background_tasks.add_task(execute_task_orchestration, resume_result["execution"])
            result["resume"] = resume_result
    return result


@router.get("/tasks/{task_id}/connector-requests", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_connector_requests(
    task_id: str,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List external connector/skill requests for a task."""
    task = _store().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return {
        "task": task,
        "connector_requests": _store().list_connector_requests(
            task_id=task_id,
            status=status,
            limit=limit,
        ),
    }


@router.patch(
    "/tasks/{task_id}/connector-requests/{request_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def update_connector_request(
    task_id: str,
    request_id: str,
    body: AgentConnectorRequestUpdate,
) -> dict[str, Any]:
    """Review or complete one external connector/skill request."""
    request = _store().get_connector_request(request_id)
    if not request or request.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail=f"Connector request '{request_id}' not found")
    status = body.status or request.get("status") or "pending_review"
    if status in {"completed", "failed", "cancelled"}:
        result = complete_connector_request(
            task_id,
            request_id,
            status=status,
            parameters=body.parameters,
            result=body.result,
            error=body.error,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Connector request '{request_id}' not found")
        return result
    updated = _store().update_connector_request(
        request_id,
        body.model_dump(exclude_unset=True),
    )
    return {"task": _store().get_task(task_id), "connector_request": updated}


@router.get("/tasks/{task_id}/usage", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_task_usage(task_id: str) -> dict[str, Any]:
    """Return usage summary for one work task."""
    task = _store().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    from core.config import get_config
    from core.database import get_usage_db

    config = get_config()
    usage_db = get_usage_db()
    return {
        "task": task,
        "summary": usage_db.get_weekly_summary(
            budget_usd=config.weekly_budget_usd,
            window_days=config.usage_window_days,
            task_id=task_id,
        ),
        "events": usage_db.list_events(task_id=task_id, window_days=config.usage_window_days),
    }


@router.post("/tasks/{task_id}/start", dependencies=[Depends(verify_api_key)], response_model=None)
async def start_task(
    task_id: str,
    body: AgentTaskStartCreate,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Plan and optionally start a tracked work task through the local AI provider."""
    try:
        result = prepare_task_orchestration(
            task_id,
            provider_id=body.provider_id,
            model=body.model,
            cwd=body.cwd,
            prompt=body.prompt,
            requested_capabilities=body.capabilities,
            auto_start=body.auto_start,
            dry_run=body.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    execution = result.get("execution")
    if body.auto_start and not body.dry_run and isinstance(execution, dict):
        background_tasks.add_task(execute_task_orchestration, execution)
    return result


@router.post("/tasks/{task_id}/resume", dependencies=[Depends(verify_api_key)], response_model=None)
async def resume_task(task_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Resume a task from its active workflow checkpoint."""
    try:
        result = resume_task_orchestration(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    execution = result.get("execution")
    if isinstance(execution, dict):
        background_tasks.add_task(execute_task_orchestration, execution)
    return result


@router.post("/tasks/{task_id}/cancel", dependencies=[Depends(verify_api_key)], response_model=None)
async def cancel_task(task_id: str) -> dict[str, Any]:
    """Cancel a tracked work task."""
    task = _store().update_task_status(task_id, "cancelled")
    if not task:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return {"task": task}


# --------------------------- Schedules ---------------------------------------
#
# Schedules are stored plans for unattended task firings. Each fire re-uses the
# exact same prepare_task_orchestration / execute_task_orchestration pipeline
# as ``POST /tasks/{task_id}/start`` — there is no separate execution path that
# bypasses policy, approvals, or audit.


@router.get("/tasks/{task_id}/schedules", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_task_schedules(task_id: str) -> dict[str, Any]:
    if _store().get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    return {"schedules": get_schedule_store().list_for_task(task_id)}


@router.post("/tasks/{task_id}/schedules", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_task_schedule(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    if _store().get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent task '{task_id}' not found")
    expression = body.get("expression")
    if not isinstance(expression, dict):
        raise HTTPException(
            status_code=400,
            detail="expression must be {'kind': 'interval'|'daily_at', ...}",
        )
    try:
        schedule = get_schedule_store().create(
            task_id=task_id,
            expression=expression,
            name=body.get("name"),
            provider_id=body.get("provider_id"),
            model=body.get("model"),
            cwd=body.get("cwd"),
            prompt=body.get("prompt"),
            capabilities=body.get("capabilities") or [],
            enabled=bool(body.get("enabled", True)),
            skip_if_active=bool(body.get("skip_if_active", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"schedule": schedule}


@router.get("/schedules", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_all_schedules(enabled_only: bool = False) -> dict[str, Any]:
    return {"schedules": get_schedule_store().list_all(enabled_only=enabled_only)}


@router.get("/schedules/{schedule_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_schedule(schedule_id: str) -> dict[str, Any]:
    schedule = get_schedule_store().get(schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found")
    return {"schedule": schedule}


@router.patch("/schedules/{schedule_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def patch_schedule(schedule_id: str, body: dict[str, Any]) -> dict[str, Any]:
    if get_schedule_store().get(schedule_id) is None:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found")
    try:
        schedule = get_schedule_store().update(schedule_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"schedule": schedule}


@router.delete("/schedules/{schedule_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def delete_schedule(schedule_id: str) -> dict[str, Any]:
    if not get_schedule_store().delete(schedule_id):
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found")
    return {"deleted": True, "schedule_id": schedule_id}


@router.post(
    "/schedules/{schedule_id}/trigger",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def trigger_schedule_now(schedule_id: str) -> dict[str, Any]:
    """Fire this schedule immediately. Goes through the same orchestrator path."""
    from agent.scheduler import _fire_schedule  # local import to avoid cycle

    schedule = get_schedule_store().get(schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found")
    await _fire_schedule(schedule)
    return {"schedule": get_schedule_store().get(schedule_id)}


@router.post("/scheduler/tick", dependencies=[Depends(verify_api_key)], response_model=None)
async def scheduler_tick_now() -> dict[str, Any]:
    """Force one scheduler tick (debug / tests). Returns count of due schedules processed."""
    fired = await get_scheduler().trigger_once()
    return {"fired": fired}


@router.get("/capabilities", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_capabilities(
    type: str | None = None,
    provider_id: str | None = None,
    status: str | None = None,
    refresh: bool = False,
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    """List durable Work Cockpit capability catalog entries."""
    store = _store()
    if refresh:
        refresh_capability_registry()
    capabilities = store.list_capabilities(
        capability_type=type,
        provider_id=provider_id,
        status=status,
        limit=limit,
    )
    if not capabilities:
        capabilities = refresh_capability_registry()
    return {"capabilities": capabilities}


@router.post("/capabilities/refresh", dependencies=[Depends(verify_api_key)], response_model=None)
async def refresh_capabilities() -> dict[str, Any]:
    """Refresh durable capability catalog from built-ins and local provider status."""
    capabilities = refresh_capability_registry()
    return {"capabilities": capabilities, "count": len(capabilities)}
