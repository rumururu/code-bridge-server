"""Agent Cockpit API routes."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from agent.agent_models import (
    AgentArtifactCreate,
    AgentConnectorRequestUpdate,
    AgentEventCreate,
    AgentRunCreate,
    AgentRunMessageCreate,
    AgentRunPreflightCreate,
    AgentTaskCreate,
    AgentTaskRunLinkCreate,
    AgentTaskStepRunCreate,
    AgentTaskStepUpdate,
    AgentTaskStartCreate,
    AgentTaskUpdate,
)
from agent.agent_store import get_agent_store
from agent.capability_registry import refresh_capability_registry
from agent.schedule_store import get_schedule_store
from agent.scheduler import get_scheduler
from agent.task_orchestrator import (
    complete_connector_request,
    execute_task_orchestration,
    execute_task_step_adapter,
    prepare_task_orchestration,
)
from agent.tool_artifacts import ARTIFACT_ROOT, record_tool_action_result
from approvals.approval_service import decide_approval
from approvals.approval_store import get_approval_store
from audit.route_audit import record_api_action
from core.database import get_project_db
from policy.policy_gate import evaluate_direct_action_gate
from terminal_action_service import execute_terminal_command_for_current_server

from .deps import verify_api_key

router = APIRouter(prefix="/api/agent", tags=["agent"])


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


@router.post("/runs", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_run(body: AgentRunCreate) -> dict[str, Any]:
    """Create a durable agent run record."""
    run = _store().create_run(
        project_name=body.project_name,
        workspace_id=body.workspace_id,
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
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """List durable agent runs."""
    return {
        "runs": _store().list_runs(
            project_name=project_name,
            workspace_id=workspace_id,
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
    if artifact.get("kind") != "device_screenshot":
        return False
    if (artifact.get("mime_type") or "").lower() != "image/png":
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
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
) -> dict[str, Any]:
    """Update one task step after parameter review."""
    step = _store().get_task_step(step_id)
    if not step or step.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail=f"Agent task step '{step_id}' not found")
    updated = _store().update_task_step(step_id, body.model_dump(exclude_unset=True))
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
