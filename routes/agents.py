"""Agent Cockpit API routes."""

import asyncio
import json
import logging
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
    enrich_draft_from_user_intent,
    get_builder_session,
)
from agent.agent_models import (
    AgentCreate,
    AgentArtifactCreate,
    AgentConnectorRequestUpdate,
    AgentDraft,
    AgentEventCreate,
    AgentUpdate,
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
from agent.agent_store import (
    AgentStoreConflictError,
    PseudoAgentProtectedError,
    add_memory_from_event,
    get_agent_store,
)
from agent.browser_action_adapter import get_browser_runtime_readiness
from agent.browser_session_store import get_browser_session_store
from agent.capability_registry import refresh_capability_registry
from agent.prompt_composer import compose_system_prompt
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
from agent.workflow_v2 import WorkflowNormalizationError, normalize_workflow
from approvals.approval_service import decide_approval
from approvals.approval_store import get_approval_store
from chat.chat_session_service import create_chat_session, get_chat_provider_selection
from audit.route_audit import record_api_action
from core.database import get_project_db
from policy.policy_gate import evaluate_direct_action_gate
from terminal_action_service import execute_terminal_command_for_current_server

from .deps import verify_api_key

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


def _agent_with_next_fire(agent: dict[str, Any]) -> dict[str, Any]:
    next_fire = compute_next_fire_at(str(agent["id"]))
    activation = _store().get_agent_activation_summary(str(agent["id"]))
    return {
        **agent,
        "next_fire_at": next_fire.isoformat() if next_fire else None,
        "activation": activation,
    }


def _resolve_dry_run_task(agent_id: str, task_id: str | None) -> dict[str, Any] | None:
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
    if not isinstance(schedule, str):
        return None
    text = schedule.strip().lower()
    if not text or text in {"now", "manual", "none", "no schedule"}:
        return None
    import re

    interval = re.search(
        r"every\s+(\d+)\s*(minute|minutes|min|m|hour|hours|hr|h|day|days|d)\b",
        text,
    )
    if interval:
        amount = int(interval.group(1))
        unit = interval.group(2)
        multiplier = 60
        if unit in {"hour", "hours", "hr", "h"}:
            multiplier = 3600
        elif unit in {"day", "days", "d"}:
            multiplier = 86400
        seconds = amount * multiplier
        return {"kind": "interval", "seconds": max(seconds, 60)}

    compact = re.fullmatch(r"every\s+(\d+)([mhd])", text)
    if compact:
        amount = int(compact.group(1))
        multiplier = {"m": 60, "h": 3600, "d": 86400}[compact.group(2)]
        return {"kind": "interval", "seconds": max(amount * multiplier, 60)}

    korean_interval = re.search(r"매\s*(\d+)\s*(분|시간|일)", text)
    if korean_interval:
        amount = int(korean_interval.group(1))
        unit = korean_interval.group(2)
        multiplier = 60
        if unit == "시간":
            multiplier = 3600
        elif unit == "일":
            multiplier = 86400
        return {"kind": "interval", "seconds": max(amount * multiplier, 60)}

    daily = re.search(r"(?:daily\s+at|daily|every\s+day\s+at)\s+(\d{1,2}:\d{2})", text)
    if daily:
        return {"kind": "daily_at", "time": daily.group(1)}
    korean_daily = re.search(r"매일\s+(\d{1,2}:\d{2})", text)
    if korean_daily:
        return {"kind": "daily_at", "time": korean_daily.group(1)}
    return None


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


def _builder_commit_intent(draft: AgentDraft, task_draft: TaskDraft | None) -> str:
    parts = [
        draft.name or "",
        draft.description or "",
        draft.system_prompt or "",
        json.dumps([step.model_dump() for step in draft.flow], ensure_ascii=False),
    ]
    if task_draft is not None:
        parts.extend(
            [
                task_draft.goal or "",
                task_draft.schedule or "",
            ]
        )
    return "\n".join(part for part in parts if isinstance(part, str) and part.strip())


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
    fallback: bool = False,
    error: str | None = None,
) -> BuilderTurnResponse:
    return BuilderTurnResponse(
        session_id=session.session_id,
        assistant_message=assistant_message,
        updated_draft=session.current_draft,
        is_ready_to_commit=session.is_ready_to_commit,
        should_offer_task=False,
        task_draft=session.task_draft,
        status=status,
        job_id=job_id,
        fallback=fallback,
        error=error,
    )


def _apply_configurator_fallback(
    session: BuilderSession,
    *,
    user_message: str,
    reason: str | None = None,
    job_id: str | None = None,
) -> BuilderTurnResponse:
    before = session.current_draft
    session.current_draft, session.task_draft = enrich_draft_from_user_intent(
        session.current_draft,
        previous_draft=before,
        task_draft=session.task_draft,
        user_message=user_message,
    )
    if _draft_has_commit_fields(session.current_draft):
        session.is_ready_to_commit = True
    message = _configurator_fallback_message(reason=reason, job_id=job_id)
    session.messages.append({"role": "assistant", "content": message})
    session.touch()
    return _builder_response(
        session,
        assistant_message=message,
        status="fallback",
        job_id=job_id,
        fallback=True,
        error=reason,
    )


def _draft_has_commit_fields(draft: AgentDraft) -> bool:
    return bool(
        (draft.name or "").strip()
        and draft.system_prompt.strip()
        and draft.provider_id
        and draft.flow
    )


def _configurator_fallback_message(
    *,
    reason: str | None = None,
    job_id: str | None = None,
) -> str:
    parts = [
        "Configurator LLM 응답이 지연되어 서버 규칙으로 먼저 저장 가능한 초안을 구성했습니다.",
        "왼쪽 Agent spec을 확인하고 필요한 세부값을 수정할 수 있습니다.",
    ]
    if job_id:
        parts.append("LLM 보강 작업은 백그라운드에서 계속 진행 중입니다.")
    if reason:
        parts.append(f"상세: {reason}")
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
        job.response = _apply_configurator_fallback(
            session,
            user_message=job.user_message,
            reason=job.error,
            job_id=job.id,
        )
        job.status = "fallback"
    except RuntimeError as exc:
        job.error = str(exc)
        job.response = _apply_configurator_fallback(
            session,
            user_message=job.user_message,
            reason=job.error,
            job_id=job.id,
        )
        job.status = "fallback"
    finally:
        job.touch()
        session.lock.release()


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
async def builder_converse(body: BuilderTurn) -> BuilderTurnResponse:
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
            return _apply_configurator_fallback(
                session,
                user_message=body.user_message,
                reason="Configurator LLM timed out before the fast response window.",
            )
        except RuntimeError as exc:
            return _apply_configurator_fallback(
                session,
                user_message=body.user_message,
                reason=str(exc),
            )

        return _apply_successful_configurator_response(
            session,
            raw_response=raw_response,
            user_message=body.user_message,
        )
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
    fallback = _apply_configurator_fallback(
        session,
        user_message=body.user_message,
        job_id=job_id,
    )
    job = BuilderConverseJob(
        id=job_id,
        session_id=session.session_id,
        user_message=body.user_message,
        status="queued",
        response=fallback,
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
async def builder_commit(body: BuilderCommitRequest) -> dict[str, Any]:
    session = get_builder_session(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Builder session '{body.session_id}' not found",
        )

    draft = body.draft
    task_draft = _resolve_commit_task_draft(body, session)
    draft, task_draft = enrich_draft_from_user_intent(
        draft,
        previous_draft=session.current_draft,
        task_draft=task_draft,
        user_message=_builder_commit_intent(draft, task_draft),
    )
    _validate_commit_draft(draft)
    if task_draft is not None:
        _validate_task_draft(task_draft)

    store = _store()
    agent = store.create_agent(
        name=(draft.name or "").strip(),
        description=draft.description,
        system_prompt=draft.system_prompt,
        provider_id=draft.provider_id,
        model=draft.model,
        tools_json=[tool.model_dump() for tool in draft.tools],
        flow_json=_normalize_agent_workflow([step.model_dump() for step in draft.flow]),
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
    if task_draft is not None:
        goal = str(task_draft.goal).strip()
        task = store.create_task(
            title=goal[:80],
            description=None,
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
        expression = _schedule_expression_from_draft(task_draft.schedule)
        if expression is not None:
            try:
                get_schedule_store().create(
                    task_id=task["id"],
                    expression=expression,
                    name=task_draft.schedule,
                    provider_id=draft.provider_id,
                    model=draft.model,
                    cwd=task_draft.cwd,
                    prompt=goal,
                    enabled=True,
                )
            except ValueError:
                # Keep commit behavior stable for free-form schedule text that
                # cannot be normalized into the current schedule expression set.
                pass
        result["task"] = task

    result["agent"] = _agent_with_next_fire(store.get_agent(agent["id"]) or agent)
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
async def create_agent(body: AgentCreate) -> dict[str, Any] | JSONResponse:
    if body.id in PSEUDO_AGENT_IDS:
        return JSONResponse(
            status_code=409,
            content={"error": "agent_id_conflict"},
        )
    agent = _store().create_agent(
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        provider_id=body.provider_id,
        model=body.model,
        tools_json=body.tools_json,
        flow_json=_normalize_agent_workflow(body.flow_json),
        policy_overrides_json=body.policy_overrides_json,
    )
    return _agent_with_next_fire(agent)


@router.get("/agents/{agent_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_agent(agent_id: str) -> dict[str, Any]:
    return _agent_with_next_fire(_require_agent(agent_id))


@router.post("/agents/{agent_id}/dry-run", dependencies=[Depends(verify_api_key)], response_model=None)
async def start_dry_run(
    agent_id: str,
    body: DryRunRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    agent = _require_agent(agent_id)
    task = _resolve_dry_run_task(agent_id, body.task_id)
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


@router.patch("/agents/{agent_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
) -> dict[str, Any] | JSONResponse:
    patch = body.model_dump(exclude_unset=True)
    if "flow_json" in patch:
        patch["flow_json"] = _normalize_agent_workflow(patch["flow_json"])
    try:
        agent = _store().update_agent(agent_id, patch)
    except PseudoAgentProtectedError:
        return _pseudo_agent_protected_response()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _agent_with_next_fire(agent)


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


@router.get(
    "/agents/{agent_id}/preview-prompt",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def preview_agent_prompt(
    agent_id: str,
    task_goal: str | None = None,
) -> dict[str, Any]:
    agent = _require_agent(agent_id)
    store = _store()
    memories = store.list_memories(agent_id, limit=100) or []
    workflow_steps = agent.get("flow_json") or []
    memory_count = store.count_memories(agent_id)
    return {
        "composed_prompt": compose_system_prompt(
            {**agent, "memories": memories},
            task_goal=task_goal,
        ),
        "memory_count": memory_count if memory_count is not None else len(memories),
        "workflow_steps": len(workflow_steps) if isinstance(workflow_steps, list) else 0,
    }


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
