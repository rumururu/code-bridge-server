"""Task orchestration service for Work Cockpit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audit.route_audit import record_api_action
from chat.chat_session_service import ChatProviderSelection, create_chat_session, get_chat_provider_selection
from chat.chat_stream_service import stream_claude_turn
from core.config import get_config
from core.runtime_paths import runtime_dir
from policy.policy_gate import evaluate_direct_action_gate
from terminal_action_service import (
    execute_terminal_command_for_current_server,
    execute_terminal_command_streaming_for_current_server,
)
from workspaces.workspace_store import get_workspace_store

from .agent_store import get_agent_store
from .capability_adapters import describe_capability_adapter
from .capability_registry import refresh_capability_registry

logger = logging.getLogger(__name__)

GLOBAL_TASK_PROJECT_NAME = "__global__"

ORCHESTRATOR_SYSTEM_PROMPT = """You are executing a Code Bridge Work Cockpit task.

Use the local project context when one is provided. Respect Code Bridge policy:
dangerous file, terminal, git, device, network, MCP, or skill actions may be
gated by approval. Keep durable outputs discoverable by summarizing what you
changed, verified, and still need from the user."""


class AgentTaskRunSink:
    """Record a background provider turn into one durable agent run."""

    def __init__(self, *, run_id: str) -> None:
        self.agent_run_id = run_id
        self.permission_required = False
        self.error_message: str | None = None

    async def send_json(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        event_type = str(data.get("type") or "background")
        if event_type == "permission_required":
            self.permission_required = True
        if event_type == "error":
            message = data.get("message")
            self.error_message = str(message) if message is not None else "Unknown error"
        provider_id = data.get("provider_id")
        get_agent_store().append_event(
            run_id=self.agent_run_id,
            event_type=event_type,
            provider_id=provider_id if isinstance(provider_id, str) else None,
            provider_event=data if event_type == "provider_event" else None,
            app_event=data if event_type != "provider_event" else data.get("normalized"),
        )


def build_task_orchestration_plan(
    task_id: str,
    *,
    provider_id: str | None = None,
    model: str | None = None,
    cwd: str | None = None,
    prompt: str | None = None,
    requested_capabilities: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build a deterministic execution plan without mutating task/run state."""
    store = get_agent_store()
    task = store.get_task(task_id)
    if not task:
        return None

    selection = _resolve_provider_selection(provider_id=provider_id, model=model)
    project_path = _resolve_project_path(task, cwd=cwd)
    workspace_id = _resolve_workspace_id(task, project_path)
    capabilities = _select_capabilities(
        task,
        provider_id=selection.provider_id,
        requested=requested_capabilities or [],
    )
    steps = _plan_steps(task, capabilities)
    launch_message = _build_launch_message(
        task,
        capabilities=capabilities,
        steps=steps,
        override_prompt=prompt,
    )
    project_name = task.get("project_name") or GLOBAL_TASK_PROJECT_NAME
    return {
        "task": task,
        "provider": {
            "provider_id": selection.provider_id,
            "provider_name": selection.provider_name,
            "model": selection.model,
        },
        "workspace_id": workspace_id,
        "project_name": project_name,
        "project_path": project_path,
        "capabilities": capabilities,
        "steps": steps,
        "launch_message": launch_message,
    }


def prepare_task_orchestration(
    task_id: str,
    *,
    provider_id: str | None = None,
    model: str | None = None,
    cwd: str | None = None,
    prompt: str | None = None,
    requested_capabilities: list[str] | None = None,
    auto_start: bool = True,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Create the durable run/steps for a task orchestration."""
    plan = build_task_orchestration_plan(
        task_id,
        provider_id=provider_id,
        model=model,
        cwd=cwd,
        prompt=prompt,
        requested_capabilities=requested_capabilities,
    )
    if plan is None or dry_run:
        return plan

    store = get_agent_store()
    task = plan["task"]
    provider = plan["provider"]
    run = store.create_run(
        project_name=None if plan["project_name"] == GLOBAL_TASK_PROJECT_NAME else plan["project_name"],
        workspace_id=plan.get("workspace_id"),
        provider_id=provider["provider_id"],
        model=provider.get("model"),
        title=f"Run task: {task.get('title')}",
        goal=task.get("goal") or task.get("title"),
        cwd=plan.get("project_path"),
        task_id=task_id,
    )
    store.add_message(
        run_id=run["id"],
        role="system",
        content=ORCHESTRATOR_SYSTEM_PROMPT,
    )
    store.add_message(
        run_id=run["id"],
        role="user",
        content=plan["launch_message"],
    )

    persisted_steps: list[dict[str, Any]] = []
    for capability in plan["capabilities"]:
        store.link_task_capability(
            task_id=task_id,
            capability_id=capability["id"],
            mode=_capability_mode(capability),
        )
    for item in plan["steps"]:
        step = store.create_task_step(
            task_id=task_id,
            run_id=run["id"],
            capability_id=item.get("capability_id"),
            title=item.get("title"),
            status=item.get("status", "queued"),
            input=item.get("input"),
            output=item.get("output"),
        )
        if step:
            persisted_steps.append(step)

    metadata = dict(task.get("metadata") or {})
    metadata["orchestrator"] = {
        "provider_id": provider["provider_id"],
        "model": provider.get("model"),
        "auto_start": auto_start,
        "run_id": run["id"],
    }
    updated_task = store.update_task(
        task_id,
        {
            "status": "queued",
            "run_id": run["id"],
            "metadata": metadata,
        },
    )
    store.append_event(
        run_id=run["id"],
        event_type="task.orchestrated",
        provider_id=provider["provider_id"],
        app_event={
            "task_id": task_id,
            "auto_start": auto_start,
            "capabilities": [capability["id"] for capability in plan["capabilities"]],
            "step_count": len(persisted_steps),
        },
    )
    return {
        **plan,
        "task": updated_task or task,
        "run": run,
        "steps": persisted_steps,
        "execution": {
            "auto_start": auto_start,
            "run_id": run["id"],
            "task_id": task_id,
            "project_name": plan["project_name"],
            "project_path": plan["project_path"],
            "provider_id": provider["provider_id"],
            "model": provider.get("model"),
            "launch_message": plan["launch_message"],
        },
    }


async def execute_task_step_adapter(
    task_id: str,
    step_id: str,
    *,
    input_override: dict[str, Any] | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | None:
    """Execute one reviewed task step when a concrete adapter is available."""
    store = get_agent_store()
    task = store.get_task(task_id)
    step = store.get_task_step(step_id)
    if not task or not step or step.get("task_id") != task_id:
        return None

    step_input = dict(step.get("input") or {})
    if input_override:
        step_input.update(input_override)
        updated = store.update_task_step(step_id, {"input": step_input})
        if updated:
            step = updated
    adapter = _step_adapter(step_input)
    run_id = step.get("run_id") or task.get("run_id")
    adapter_name = adapter.get("adapter")
    invocation = adapter.get("invocation")
    policy_operation = str(adapter.get("policy_operation") or "")

    store.update_task_step(step_id, {"status": "running"})
    if run_id:
        store.append_event(
            run_id=run_id,
            event_type="task.step.started",
            app_event={"task_id": task_id, "step_id": step_id, "adapter": adapter},
        )

    if adapter_name == "codebridge_builtin" and policy_operation == "file.read":
        return _complete_step(
            task=task,
            step_id=step_id,
            run_id=run_id,
            output=_execute_file_read_step(task, step_input),
        )

    if adapter_name == "codebridge_builtin" and policy_operation == "process.terminal":
        commands = _commands_from_step_input(step_input)
        if not commands:
            return _block_step(
                task=task,
                step_id=step_id,
                run_id=run_id,
                reason="Step input requires a command or commands list.",
                adapter=adapter,
            )
        project_name = task.get("project_name")
        if not isinstance(project_name, str) or not project_name:
            return _block_step(
                task=task,
                step_id=step_id,
                run_id=run_id,
                reason="Terminal steps require a project-linked task.",
                adapter=adapter,
            )
        timeout = int(step_input.get("timeout") or 300)
        gate = evaluate_direct_action_gate(
            operation="process.terminal",
            project_name=project_name,
            run_id=run_id if isinstance(run_id, str) else None,
            details={"commands": commands, "timeout": timeout, "step_id": step_id},
            require_approval=require_approval,
            approval_id=approval_id,
        )
        if not gate["allowed"]:
            blocked = _block_step(
                task=task,
                step_id=step_id,
                run_id=run_id,
                reason="Approval required before running terminal step.",
                adapter=adapter,
                extra={"approval": gate["payload"], "approval_required": True},
            )
            blocked["approval_required"] = True
            blocked["approval"] = gate["payload"]
            return blocked

        results: list[dict[str, Any]] = []
        success = True
        for command in commands:
            # Stream stdout/stderr chunks as step.log events so the Cockpit can
            # show a build's progress in near-real-time over /ws/agent/runs/.
            # The agent run WS poll interval is 1 second, so the user sees
            # logs within ~1 s of them being printed.
            emitter = _make_step_log_emitter(
                run_id=run_id if isinstance(run_id, str) else None,
                task_id=task_id,
                step_id=step_id,
                command=command,
            )
            result = await execute_terminal_command_streaming_for_current_server(
                project_name,
                command=command,
                timeout=timeout,
                on_chunk=emitter,
            )
            record_api_action(
                operation="process.terminal",
                project_name=project_name,
                run_id=run_id if isinstance(run_id, str) else None,
                details={"command": command, "timeout": timeout, "step_id": step_id},
                success=result.success,
                status_code=result.status_code,
            )
            results.append(
                {
                    "command": command,
                    "success": result.success,
                    "status_code": result.status_code,
                    "payload": result.payload,
                }
            )
            success = success and result.success
            if not result.success:
                break
        if success:
            return _complete_step(
                task=task,
                step_id=step_id,
                run_id=run_id,
                output={"commands": results},
            )
        return _fail_step(
            task=task,
            step_id=step_id,
            run_id=run_id,
            error={"commands": results},
        )

    if invocation in {"deferred_skill", "deferred_mcp", "provider_permission"}:
        connector_request = store.create_connector_request(
            task_id=task_id,
            step_id=step_id,
            run_id=run_id if isinstance(run_id, str) else None,
            connector_type=str(adapter_name or "connector"),
            name=str(adapter.get("policy_operation") or adapter.get("adapter") or "connector"),
            status="pending_review",
            adapter=adapter,
            parameters=step_input,
        )
        return _block_step(
            task=task,
            step_id=step_id,
            run_id=run_id,
            reason="This adapter requires parameter review before direct invocation.",
            adapter=adapter,
            extra={"review_required": True, "connector_request": connector_request},
        )

    return _block_step(
        task=task,
        step_id=step_id,
        run_id=run_id,
        reason="No direct executor is registered for this step adapter.",
        adapter=adapter,
    )


async def execute_task_orchestration(execution: dict[str, Any]) -> None:
    """Run the prepared task through the selected local CLI provider."""
    store = get_agent_store()
    task_id = str(execution["task_id"])
    run_id = str(execution["run_id"])
    provider_id = str(execution["provider_id"])
    model = execution.get("model") if isinstance(execution.get("model"), str) else None
    project_name = str(execution.get("project_name") or GLOBAL_TASK_PROJECT_NAME)
    project_path = str(execution.get("project_path") or _global_task_path())
    launch_message = str(execution.get("launch_message") or "")

    steps = store.list_task_steps(task_id)
    execute_step = _find_step(steps, "Execute")
    summary_step = _find_step(steps, "Summarize")
    store.update_run_status(run_id, "running")
    store.update_task(task_id, {"status": "running"})
    if execute_step:
        store.update_task_step(execute_step["id"], {"status": "running"})
    store.append_event(
        run_id=run_id,
        event_type="task.execution.started",
        provider_id=provider_id,
        app_event={"task_id": task_id, "project_path": project_path},
    )

    sink = AgentTaskRunSink(run_id=run_id)
    try:
        selection = ChatProviderSelection(
            provider_id=provider_id,
            provider_name=_provider_name(provider_id),
            model=model,
        )
        session = await create_chat_session(
            project_name=project_name,
            project_path=project_path,
            selection=selection,
        )
        completed = await stream_claude_turn(
            sink,
            session,
            project_name=project_name,
            user_message=launch_message,
        )
    except Exception as exc:
        logger.exception("task orchestration failed task_id=%s run_id=%s", task_id, run_id)
        _finish_execution(
            task_id=task_id,
            run_id=run_id,
            execute_step=execute_step,
            summary_step=summary_step,
            status="failed",
            error={"message": str(exc)},
        )
        return

    if sink.permission_required:
        _finish_execution(
            task_id=task_id,
            run_id=run_id,
            execute_step=execute_step,
            summary_step=summary_step,
            status="blocked",
            error={"message": "Waiting for approval."},
        )
        return
    if sink.error_message or not completed:
        _finish_execution(
            task_id=task_id,
            run_id=run_id,
            execute_step=execute_step,
            summary_step=summary_step,
            status="failed",
            error={"message": sink.error_message or "Provider turn did not complete."},
        )
        return

    _finish_execution(
        task_id=task_id,
        run_id=run_id,
        execute_step=execute_step,
        summary_step=summary_step,
        status="completed",
        result={"message": "Provider turn completed."},
    )


def _finish_execution(
    *,
    task_id: str,
    run_id: str,
    execute_step: dict[str, Any] | None,
    summary_step: dict[str, Any] | None,
    status: str,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    store = get_agent_store()
    store.update_run_status(run_id, status)
    updates: dict[str, Any] = {"status": status}
    if result is not None:
        updates["result"] = result
    if error is not None:
        updates["error"] = error
    store.update_task(task_id, updates)
    step_status = status if status != "blocked" else "blocked"
    if execute_step:
        store.update_task_step(
            execute_step["id"],
            {
                "status": step_status,
                "output": result or error or {},
            },
        )
    if summary_step and status == "completed":
        store.update_task_step(
            summary_step["id"],
            {
                "status": "completed",
                "output": result or {},
            },
        )
    store.append_event(
        run_id=run_id,
        event_type=f"task.execution.{status}",
        app_event={"task_id": task_id, "result": result or {}, "error": error or {}},
    )


def _complete_step(
    *,
    task: dict[str, Any],
    step_id: str,
    run_id: str | None,
    output: dict[str, Any],
) -> dict[str, Any]:
    store = get_agent_store()
    step = store.update_task_step(step_id, {"status": "completed", "output": output})
    if run_id:
        store.append_event(
            run_id=run_id,
            event_type="task.step.completed",
            app_event={"task_id": task["id"], "step_id": step_id, "output": output},
        )
    return {"task": store.get_task(task["id"]), "step": step, "output": output}


def _block_step(
    *,
    task: dict[str, Any],
    step_id: str,
    run_id: str | None,
    reason: str,
    adapter: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = {"reason": reason, "adapter": adapter}
    if extra:
        output.update(extra)
    store = get_agent_store()
    step = store.update_task_step(step_id, {"status": "blocked", "output": output})
    if run_id:
        store.append_event(
            run_id=run_id,
            event_type="task.step.blocked",
            app_event={"task_id": task["id"], "step_id": step_id, "output": output},
        )
    return {"task": store.get_task(task["id"]), "step": step, "output": output}


def _fail_step(
    *,
    task: dict[str, Any],
    step_id: str,
    run_id: str | None,
    error: dict[str, Any],
) -> dict[str, Any]:
    store = get_agent_store()
    step = store.update_task_step(step_id, {"status": "failed", "output": error})
    if run_id:
        store.append_event(
            run_id=run_id,
            event_type="task.step.failed",
            app_event={"task_id": task["id"], "step_id": step_id, "error": error},
        )
    return {"task": store.get_task(task["id"]), "step": step, "error": error}


def complete_connector_request(
    task_id: str,
    request_id: str,
    *,
    status: str = "completed",
    parameters: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Complete a connector request and synchronize its task step."""
    store = get_agent_store()
    request = store.get_connector_request(request_id)
    if not request or request.get("task_id") != task_id:
        return None
    updates: dict[str, Any] = {"status": status}
    if parameters is not None:
        updates["parameters"] = parameters
    if result is not None:
        updates["result"] = result
    if error is not None:
        updates["error"] = error
    updated = store.update_connector_request(request_id, updates)
    step_id = request.get("step_id")
    step = None
    if isinstance(step_id, str) and step_id:
        step_status = "completed" if status == "completed" else "failed" if status == "failed" else "blocked"
        step = store.update_task_step(
            step_id,
            {
                "status": step_status,
                "output": {
                    "connector_request": updated,
                    "result": result or {},
                    "error": error or {},
                },
            },
        )
    run_id = request.get("run_id")
    if isinstance(run_id, str) and run_id:
        store.append_event(
            run_id=run_id,
            event_type=f"connector_request.{status}",
            app_event={
                "task_id": task_id,
                "step_id": step_id,
                "connector_request": updated,
            },
        )
    return {
        "task": store.get_task(task_id),
        "connector_request": updated,
        "step": step,
    }


def _make_step_log_emitter(
    *,
    run_id: str | None,
    task_id: str,
    step_id: str,
    command: str,
):
    """Build an ``on_chunk`` callback that appends ``step.log`` events.

    Chunks are coalesced per stream (stdout / stderr) so we emit one event
    per line, not one event per 4 KB read. This keeps the agent_events table
    tidy while still giving the Cockpit near-real-time progress.

    When ``run_id`` is missing (e.g. ad-hoc step run without a run binding)
    the emitter returns a no-op so the caller doesn't have to branch.
    """
    if not run_id:
        return None

    store = get_agent_store()
    buffers: dict[str, str] = {"stdout": "", "stderr": ""}

    def _flush(stream: str) -> None:
        line = buffers[stream]
        if not line:
            return
        buffers[stream] = ""
        try:
            store.append_event(
                run_id=run_id,
                event_type="step.log",
                app_event={
                    "task_id": task_id,
                    "step_id": step_id,
                    "command": command,
                    "stream": stream,
                    "data": line,
                },
            )
        except Exception:  # noqa: BLE001 — never break terminal execution
            logger.exception("failed to append step.log event")

    def emit(chunk: dict[str, Any]) -> None:
        stream = chunk.get("stream") if isinstance(chunk, dict) else None
        data = chunk.get("data") if isinstance(chunk, dict) else None
        if stream not in {"stdout", "stderr"} or not isinstance(data, str):
            return
        buffers[stream] += data
        while "\n" in buffers[stream]:
            line, _, rest = buffers[stream].partition("\n")
            buffers[stream] = rest
            try:
                store.append_event(
                    run_id=run_id,
                    event_type="step.log",
                    app_event={
                        "task_id": task_id,
                        "step_id": step_id,
                        "command": command,
                        "stream": stream,
                        "data": line,
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to append step.log event")
        # Flush a trailing partial line if it grows large enough — typical
        # progress bars (`\r`-overwritten lines) never see `\n`, so without
        # this they would be invisible until completion.
        if len(buffers[stream]) >= 256:
            _flush(stream)

    return emit


def _step_adapter(step_input: dict[str, Any]) -> dict[str, Any]:
    adapter = step_input.get("adapter")
    return adapter if isinstance(adapter, dict) else {}


def _commands_from_step_input(step_input: dict[str, Any]) -> list[str]:
    raw_commands = step_input.get("commands")
    if isinstance(raw_commands, list):
        return [str(command).strip() for command in raw_commands if str(command).strip()]
    raw_command = step_input.get("command")
    if isinstance(raw_command, str) and raw_command.strip():
        return [raw_command.strip()]
    return []


def _execute_file_read_step(task: dict[str, Any], step_input: dict[str, Any]) -> dict[str, Any]:
    root = Path(_resolve_project_path(task, cwd=None)).resolve()
    raw_paths = step_input.get("paths")
    if not isinstance(raw_paths, list):
        raw_path = step_input.get("path")
        raw_paths = [raw_path] if isinstance(raw_path, str) else []
    max_chars = int(step_input.get("max_chars") or 12000)
    files: list[dict[str, Any]] = []
    for item in raw_paths:
        if not isinstance(item, str) or not item.strip():
            continue
        candidate = (root / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            files.append({"path": item, "readable": False, "reason": "path escapes workspace"})
            continue
        if not candidate.is_file():
            files.append({"path": item, "readable": False, "reason": "file not found"})
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        files.append(
            {
                "path": str(candidate),
                "readable": True,
                "content": text[:max_chars] if truncated else text,
                "truncated": truncated,
                "bytes": candidate.stat().st_size,
            }
        )
    return {"files": files}


def _resolve_provider_selection(
    *,
    provider_id: str | None,
    model: str | None,
) -> ChatProviderSelection:
    if provider_id:
        return ChatProviderSelection(
            provider_id=provider_id,
            provider_name=_provider_name(provider_id),
            model=model,
        )
    selection = get_chat_provider_selection()
    if model:
        return ChatProviderSelection(
            provider_id=selection.provider_id,
            provider_name=selection.provider_name,
            model=model,
        )
    return selection


def _resolve_project_path(task: dict[str, Any], *, cwd: str | None) -> str:
    if cwd:
        return cwd
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        for key in ("root_path", "project_path", "cwd", "path"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    workspace_id = task.get("workspace_id")
    if isinstance(workspace_id, str) and workspace_id:
        workspace = get_workspace_store().get_workspace(workspace_id)
        root_path = workspace.get("root_path") if workspace else None
        if isinstance(root_path, str) and root_path:
            return root_path
    project_name = task.get("project_name")
    if isinstance(project_name, str) and project_name:
        project = get_config().get_project(project_name)
        project_path = project.get("path") if isinstance(project, dict) else None
        if isinstance(project_path, str) and project_path:
            return project_path
    return _global_task_path()


def _resolve_workspace_id(task: dict[str, Any], project_path: str) -> str | None:
    workspace_id = task.get("workspace_id")
    if isinstance(workspace_id, str) and workspace_id:
        return workspace_id
    project_name = task.get("project_name")
    if not isinstance(project_name, str) or not project_name:
        return None
    workspace = get_workspace_store().get_or_create_project_workspace(
        project_name=project_name,
        root_path=project_path,
        display_name=project_name,
    )
    return workspace.get("id") if isinstance(workspace.get("id"), str) else None


def _select_capabilities(
    task: dict[str, Any],
    *,
    provider_id: str,
    requested: list[str],
) -> list[dict[str, Any]]:
    catalog = refresh_capability_registry()
    selected: list[dict[str, Any]] = []

    def add_by(predicate) -> None:
        for capability in catalog:
            if predicate(capability):
                _append_unique(selected, capability)
                return

    add_by(lambda cap: cap.get("type") == "llm_cli" and cap.get("provider_id") == provider_id)
    for item in requested:
        normalized = item.strip()
        if not normalized:
            continue
        add_by(lambda cap, normalized=normalized: cap.get("id") == normalized or cap.get("name") == normalized)

    kind = str(task.get("kind") or "general").lower()
    labels = {str(label).lower() for label in task.get("labels") or []}
    text = " ".join(
        str(value or "").lower()
        for value in (task.get("title"), task.get("goal"), task.get("description"), kind, " ".join(labels))
    )
    add_by(lambda cap: cap.get("type") == "builtin" and cap.get("name") == "file.read")
    if kind == "app_build" or "app" in labels:
        for name in ("app_builder", "file.write", "process.terminal", "device.control"):
            add_by(lambda cap, name=name: cap.get("type") == "builtin" and cap.get("name") == name)
    if kind in {"review", "ops"} or "git" in text:
        add_by(lambda cap: cap.get("type") == "builtin" and cap.get("name") == "git")
    if kind in {"ops", "review"} or any(token in text for token in ("build", "test", "run", "deploy")):
        add_by(lambda cap: cap.get("type") == "builtin" and cap.get("name") == "process.terminal")
    if kind == "research" or any(token in text for token in ("research", "web", "browser", "search")):
        for name in ("browser", "documents"):
            add_by(lambda cap, name=name: cap.get("type") == "skill" and cap.get("name") == name)
    if any(token in text for token in ("github", "issue", "pr", "pull request")):
        add_by(lambda cap: cap.get("type") == "mcp_server" and cap.get("name") == "github")
    if any(token in text for token in ("gmail", "email", "mail")):
        add_by(lambda cap: cap.get("type") == "mcp_server" and cap.get("name") == "gmail")
    return selected


def _plan_steps(task: dict[str, Any], capabilities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    llm_cap = _first_capability(capabilities, "llm_cli")
    steps.append(
        {
            "title": "Prepare task context",
            "capability_id": llm_cap.get("id") if llm_cap else None,
            "status": "queued",
            "input": {"task_id": task["id"], "kind": task.get("kind")},
        }
    )
    for capability in capabilities:
        if capability.get("type") == "llm_cli":
            continue
        steps.append(
            {
                "title": f"Use {capability.get('name')}",
                "capability_id": capability.get("id"),
                "status": "queued",
                "input": {
                    "capability": capability.get("name"),
                    "type": capability.get("type"),
                    "permission_level": capability.get("permission_level"),
                    "adapter": describe_capability_adapter(capability),
                },
            }
        )
    steps.append(
        {
            "title": "Execute provider turn",
            "capability_id": llm_cap.get("id") if llm_cap else None,
            "status": "queued",
            "input": {"provider": llm_cap.get("provider_id") if llm_cap else None},
        }
    )
    steps.append(
        {
            "title": "Summarize result",
            "capability_id": llm_cap.get("id") if llm_cap else None,
            "status": "queued",
        }
    )
    return steps


def _build_launch_message(
    task: dict[str, Any],
    *,
    capabilities: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    override_prompt: str | None,
) -> str:
    if override_prompt and override_prompt.strip():
        user_goal = override_prompt.strip()
    else:
        user_goal = str(task.get("goal") or task.get("description") or task.get("title") or "").strip()
    capability_lines = [
        f"- {capability.get('type')}:{capability.get('name')} status={capability.get('status')}"
        for capability in capabilities
    ]
    step_lines = [f"{index + 1}. {step.get('title')}" for index, step in enumerate(steps)]
    return "\n".join(
        [
            ORCHESTRATOR_SYSTEM_PROMPT,
            "",
            "Task:",
            f"- id: {task.get('id')}",
            f"- title: {task.get('title')}",
            f"- kind: {task.get('kind')}",
            f"- project: {task.get('project_name') or 'none'}",
            "",
            "Goal:",
            user_goal,
            "",
            "Available Code Bridge capabilities selected for this task:",
            *capability_lines,
            "",
            "Expected execution outline:",
            *step_lines,
            "",
            "When finished, summarize changes, checks, artifacts, and any required follow-up.",
        ]
    )


def _append_unique(items: list[dict[str, Any]], capability: dict[str, Any]) -> None:
    capability_id = capability.get("id")
    if not isinstance(capability_id, str):
        return
    if any(item.get("id") == capability_id for item in items):
        return
    items.append(capability)


def _first_capability(capabilities: list[dict[str, Any]], capability_type: str) -> dict[str, Any] | None:
    for capability in capabilities:
        if capability.get("type") == capability_type:
            return capability
    return None


def _find_step(steps: list[dict[str, Any]], title_prefix: str) -> dict[str, Any] | None:
    for step in steps:
        title = step.get("title")
        if isinstance(title, str) and title.startswith(title_prefix):
            return step
    return None


def _capability_mode(capability: dict[str, Any]) -> str:
    permission = str(capability.get("permission_level") or "")
    if permission in {"approval", "desktop_only"}:
        return "execute"
    if capability.get("type") in {"skill", "mcp_server", "llm_cli"}:
        return "execute"
    return "read"


def _provider_name(provider_id: str) -> str:
    return {
        "anthropic": "Claude",
        "openai": "Codex",
        "google": "Gemini",
    }.get(provider_id, provider_id.title())


def _global_task_path() -> str:
    return str(runtime_dir("global_chat", Path.home() / ".code-bridge" / "global_chat"))
