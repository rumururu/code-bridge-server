"""Task orchestration service for Work Cockpit."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audit.route_audit import record_api_action
from chat.chat_session_service import ChatProviderSelection, create_chat_session, get_chat_provider_selection
from chat.chat_stream_service import stream_claude_turn
from core.config import get_config
from core.runtime_paths import runtime_dir
from policy.policy_gate import evaluate_direct_action_gate
from terminal_action_service import (
    execute_terminal_command_streaming_for_current_server,
)
from workspaces.workspace_store import get_workspace_store

from .agent_store import get_agent_store
from .app_action_adapter import AppActionAdapterResult
from .app_action_executor import execute_app_actions
from .browser_action_adapter import BrowserActionAdapterResult
from .browser_action_executor import execute_browser_actions
from .browser_runtime_manager import get_browser_runtime_manager
from .browser_session_store import get_browser_session_store
from .capability_adapters import describe_capability_adapter
from .capability_registry import refresh_capability_registry
from .prompt_composer import compose_system_prompt
from .workflow_v2 import WorkflowNormalizationError, normalize_workflow

logger = logging.getLogger(__name__)

GLOBAL_TASK_PROJECT_NAME = "__global__"

ORCHESTRATOR_SYSTEM_PROMPT = """You are executing a Code Bridge Work Cockpit task.

Use the local project context when one is provided. Respect Code Bridge policy:
dangerous file, terminal, git, device, network, MCP, or skill actions may be
gated by approval. Keep durable outputs discoverable by summarizing what you
changed, verified, and still need from the user."""


#: Conditional-language heuristics used to flag an LLM reasoning event as a
#: branch precursor. When a matching reasoning event is followed by a
#: ``tool_use`` event, ``AgentTaskRunSink`` self-injects a
#: ``decision_marker`` event so the Flutter flow renderer can pivot the
#: timeline into a 2-column branch view.
#:
#: The patterns intentionally favor *recall* over *precision*: the worst-case
#: false-positive is a redundant diamond node in the UI (DESIGN
#: § 4.2 Case b), never a broken run. Heuristic refinement is open work
#: tracked in TASK_006.
_DECISION_PATTERNS = (
    re.compile(r"\bif\s+\w", re.IGNORECASE),
    re.compile(r"\botherwise\b", re.IGNORECASE),
    re.compile(r"\bdepending on\b", re.IGNORECASE),
    re.compile(r"\belse\s", re.IGNORECASE),
    re.compile(r"만약"),
    re.compile(r"결과에 따라"),
    re.compile(r"조건"),
)


class AgentTaskRunSink:
    """Record a background provider turn into one durable agent run.

    Beyond the raw passthrough that TASK_001 established, this sink also
    looks ahead one event to detect "LLM reasons about a branch, then
    calls a tool" patterns. When matched, a synthetic ``decision_marker``
    event is inserted before the upcoming ``tool_use`` and the
    ``tool_use`` row's ``parent_event_id`` is linked to that marker so
    the Flutter renderer can group children under a diamond gateway.
    """

    def __init__(self, *, run_id: str) -> None:
        self.agent_run_id = run_id
        self.permission_required = False
        self.error_message: str | None = None
        # What the model actually said. Without it a completed LLM step reads
        # "Workflow step completed." and the answer — the diagnosis you asked
        # it for — is only in the event log.
        self.result_text: str | None = None
        # decision_marker emit state (TASK_005)
        self._last_reasoning_event_id: str | None = None
        self._last_reasoning_decision_text: str | None = None

    async def send_json(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        event_type = str(data.get("type") or "background")
        if event_type == "permission_required":
            self.permission_required = True
        if event_type == "error":
            message = data.get("message")
            self.error_message = str(message) if message is not None else "Unknown error"
        if event_type == "complete":
            content = data.get("content")
            if isinstance(content, str) and content.strip():
                self.result_text = content.strip()
        elif event_type == "result":
            content = data.get("result")
            if isinstance(content, str) and content.strip():
                self.result_text = content.strip()

        provider_id_raw = data.get("provider_id")
        provider_id = provider_id_raw if isinstance(provider_id_raw, str) else None
        call_id_raw = data.get("call_id")
        call_id = call_id_raw if isinstance(call_id_raw, str) else None
        parent_event_id_raw = data.get("parent_event_id")
        parent_event_id = (
            parent_event_id_raw if isinstance(parent_event_id_raw, str) else None
        )

        store = get_agent_store()

        # --- Decision-marker injection (TASK_005) -----------------------------
        # If the *previous* assistant/output event contained a branching
        # phrase and the current event is a ``tool_use`` from the same
        # provider, slip a ``decision_marker`` row in front of the tool
        # call and rewire the tool call's parent to the marker.
        if (
            event_type in {"assistant", "output"}
            and self._is_tool_use(data)
            and self._last_reasoning_decision_text
        ):
            marker = store.append_event(
                run_id=self.agent_run_id,
                event_type="decision_marker",
                provider_id=provider_id,
                app_event={
                    "condition_text": self._last_reasoning_decision_text,
                    "decision_label": self._summarize_condition(
                        self._last_reasoning_decision_text
                    ),
                },
                parent_event_id=self._last_reasoning_event_id,
            )
            marker_id = marker.get("id") if isinstance(marker, dict) else None
            # Consume the cached snippet so we don't double-fire on the
            # next tool_use in the same turn.
            self._last_reasoning_decision_text = None

            # Now persist the tool_use itself, linked back to the marker
            # we just emitted. Caller-provided parent_event_id wins (rare;
            # nothing today sets it on raw provider events).
            store.append_event(
                run_id=self.agent_run_id,
                event_type=event_type,
                provider_id=provider_id,
                provider_event=data if event_type == "provider_event" else None,
                app_event=data if event_type != "provider_event" else data.get("normalized"),
                call_id=call_id,
                parent_event_id=parent_event_id or marker_id,
            )
            return

        # --- Normal append path (unchanged from TASK_001) ---------------------
        stored = store.append_event(
            run_id=self.agent_run_id,
            event_type=event_type,
            provider_id=provider_id,
            provider_event=data if event_type == "provider_event" else None,
            app_event=data if event_type != "provider_event" else data.get("normalized"),
            call_id=call_id,
            parent_event_id=parent_event_id,
        )

        # --- Reasoning cache update (TASK_005) --------------------------------
        # Track the last assistant/output text event so the next tool_use
        # knows whether to wedge a decision_marker in. Benign intermediate
        # events (background, status, etc.) leave the cache alone so a
        # later tool_use still pairs with the most recent branching
        # reasoning (DESIGN § 3.3 step 3). Hard error / approval events
        # discard the cache so we never retroactively span a control
        # boundary.
        if event_type in {"assistant", "output"}:
            text = self._extract_assistant_text(data)
            if text:
                if any(p.search(text) for p in _DECISION_PATTERNS):
                    self._last_reasoning_decision_text = self._extract_decision_snippet(text)
                else:
                    # Replacement, not append: only the most recent
                    # reasoning influences the next tool_use.
                    self._last_reasoning_decision_text = None
                if isinstance(stored, dict):
                    stored_id = stored.get("id")
                    if isinstance(stored_id, str):
                        self._last_reasoning_event_id = stored_id
        elif event_type in {"error", "permission_required", "result"}:
            # Hard control-flow boundaries invalidate the cache. A
            # decision_marker that spans an error/approval is misleading
            # to the user (the branch didn't actually take, the run
            # paused).
            self._last_reasoning_decision_text = None

    # ------------------------------------------------------------------
    # Decision-marker helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_assistant_text(data: dict[str, Any]) -> str:
        """Return concatenated text from an assistant/output envelope.

        Supports both Anthropic-style ``content`` blocks with ``type ==
        "text"`` and Codex-style ``content`` blocks with ``type ==
        "output_text"``. Anything else (tool_use / tool_result / etc.)
        is skipped — those never carry branching reasoning.
        """
        msg = data.get("message")
        if not isinstance(msg, dict):
            return ""
        content = msg.get("content")
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"text", "output_text"}:
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return " ".join(parts)

    @staticmethod
    def _is_tool_use(data: dict[str, Any]) -> bool:
        msg = data.get("message")
        if not isinstance(msg, dict):
            return False
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            return False
        first = content[0]
        return isinstance(first, dict) and first.get("type") == "tool_use"

    @staticmethod
    def _extract_decision_snippet(text: str, max_chars: int = 160) -> str:
        """Slice the first conditional sentence out of an LLM reasoning blob.

        We anchor the snippet on the first matched pattern so the UI label
        stays close to "if foo then bar" rather than echoing the whole
        reasoning paragraph.
        """
        for pattern in _DECISION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            start = max(0, match.start() - 20)
            end = min(len(text), match.start() + max_chars)
            snippet = text[start:end].strip()
            return snippet
        return text[:max_chars].strip()

    @staticmethod
    def _summarize_condition(snippet: str, max_chars: int = 60) -> str:
        """Render a single short label for the diamond node body."""
        cleaned = " ".join(snippet.split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"


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
    workflow_steps = _workflow_steps_for_task(task)
    capabilities = _select_capabilities(
        task,
        provider_id=selection.provider_id,
        requested=requested_capabilities or [],
        workflow_steps=workflow_steps,
    )
    steps = _plan_steps(task, capabilities, workflow_steps=workflow_steps)
    task_goal = _task_goal_text(task, prompt)
    system_prompt = _compose_assigned_agent_prompt(task, task_goal=task_goal) or ORCHESTRATOR_SYSTEM_PROMPT
    launch_message = _build_launch_message(
        task,
        capabilities=capabilities,
        steps=steps,
        user_goal=task_goal,
        system_prompt=system_prompt,
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
        "system_prompt": system_prompt,
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
        agent_id=task.get("assigned_agent_id") or "agent_legacy_chat",
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
        content=str(plan.get("system_prompt") or ORCHESTRATOR_SYSTEM_PROMPT),
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


def resume_task_orchestration(task_id: str) -> dict[str, Any] | None:
    """Build execution payload for resuming a task from its active checkpoint."""
    store = get_agent_store()
    checkpoint = store.get_task_checkpoint(task_id)
    if checkpoint is None:
        return None
    if checkpoint.get("checkpoint") is None or checkpoint.get("step") is None:
        raise ValueError("Task has no active checkpoint to resume.")
    task = checkpoint.get("task")
    run = checkpoint.get("run")
    if not isinstance(task, dict) or not isinstance(run, dict):
        raise ValueError("Checkpoint is missing task or run context.")
    if run.get("status") in {"completed", "cancelled"}:
        raise ValueError("Completed or cancelled runs cannot be resumed.")

    run_id = str(run["id"])
    provider_id = str(run.get("provider_id") or "openai")
    project_name = str(run.get("project_name") or GLOBAL_TASK_PROJECT_NAME)
    project_path = str(run.get("cwd") or _resolve_project_path(task, cwd=None))
    messages = store.list_messages(run_id)
    launch_message = _resume_launch_message(task, messages)
    store.append_event(
        run_id=run_id,
        event_type="task.execution.resume_requested",
        provider_id=provider_id,
        app_event={
            "task_id": task_id,
            "step_id": checkpoint["step"]["id"],
            "checkpoint": checkpoint.get("checkpoint"),
        },
    )
    return {
        "task": task,
        "run": run,
        "checkpoint": checkpoint,
        "execution": {
            "auto_start": True,
            "resume": True,
            "run_id": run_id,
            "task_id": task_id,
            "project_name": project_name,
            "project_path": project_path,
            "provider_id": provider_id,
            "model": run.get("model"),
            "launch_message": launch_message,
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
    if step_input.get("workflow_step_id"):
        return await _execute_single_workflow_task_step(
            task=task,
            step=step,
            step_input=step_input,
        )
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


async def _execute_single_workflow_task_step(
    *,
    task: dict[str, Any],
    step: dict[str, Any],
    step_input: dict[str, Any],
) -> dict[str, Any]:
    """Execute exactly one persisted workflow-backed task step.

    Full task execution walks the workflow and applies transitions. This helper
    is intentionally single-step so UI can offer "run this step" while the user
    is reviewing or debugging a workflow.
    """

    store = get_agent_store()
    task_id = str(task["id"])
    step_id = str(step["id"])
    run_id = str(step.get("run_id") or task.get("run_id") or "")
    run = store.get_run(run_id) if run_id else None
    provider_id = str(
        (run or {}).get("provider_id")
        or task.get("provider_id")
        or "openai"
    )
    model = (run or {}).get("model")
    model = model if isinstance(model, str) else None
    project_name = str(
        (run or {}).get("project_name")
        or task.get("project_name")
        or GLOBAL_TASK_PROJECT_NAME
    )
    project_path = str(
        (run or {}).get("cwd")
        or _resolve_project_path(task, cwd=None)
    )
    launch_message = str(
        task.get("goal")
        or task.get("description")
        or task.get("title")
        or "Run this workflow step."
    )
    workflow_type = str(step_input.get("workflow_type") or "llm")

    if not run_id:
        return _block_step(
            task=task,
            step_id=step_id,
            run_id=None,
            reason="Workflow step execution requires a linked run.",
            adapter={"adapter": "workflow", "workflow_type": workflow_type},
        )

    if workflow_type == "browser_action":
        completed = await _execute_browser_action_workflow_step(
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            project_name=project_name,
            project_path=project_path,
            step=step,
        )
        if completed is None:
            return {
                "task": store.get_task(task_id),
                "step": store.get_task_step(step_id),
                "status": "waiting_for_user",
            }
        return {
            "task": store.get_task(task_id),
            "step": store.get_task_step(step_id),
            "status": "completed" if completed else "failed",
        }
    if _is_app_action_workflow_type(workflow_type):
        completed = await _execute_app_action_workflow_step(
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            project_name=project_name,
            project_path=project_path,
            step=step,
        )
        if completed is None:
            return {
                "task": store.get_task(task_id),
                "step": store.get_task_step(step_id),
                "status": "waiting_for_user",
            }
        return {
            "task": store.get_task(task_id),
            "step": store.get_task_step(step_id),
            "status": "completed" if completed else "failed",
        }

    if workflow_type == "shell":
        completed = await _execute_shell_workflow_step(
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            project_path=project_path,
            step=step,
        )
        return {
            "task": store.get_task(task_id),
            "step": store.get_task_step(step_id),
            "status": "completed" if completed else "failed",
        }

    if workflow_type in {"manual_handoff", "mcp_tool", "approval_gate"}:
        _wait_for_user_step(
            task_id=task_id,
            run_id=run_id,
            step=step,
            reason=workflow_type,
        )
        return {
            "task": store.get_task(task_id),
            "step": store.get_task_step(step_id),
            "status": "waiting_for_user",
        }

    if workflow_type == "condition":
        return _complete_step(
            task=task,
            step_id=step_id,
            run_id=run_id,
            output={"result": "condition step completed without branching"},
        )

    completed = await _execute_llm_workflow_step(
        task_id=task_id,
        run_id=run_id,
        provider_id=provider_id,
        model=model,
        project_name=project_name,
        project_path=project_path,
        launch_message=launch_message,
        step=step,
        previous_steps=_previous_completed_workflow_steps(
            store.list_task_steps(task_id),
            current_step=step,
        ),
    )
    return {
        "task": store.get_task(task_id),
        "step": store.get_task_step(step_id),
        "status": "completed" if completed else "failed",
    }


async def execute_task_orchestration(execution: dict[str, Any]) -> None:
    """Run the prepared task through the selected local CLI provider."""
    store = get_agent_store()
    if _execution_is_dry_run(execution, store=store):
        await _simulate_run(execution)
        return

    task_id = str(execution["task_id"])
    run_id = str(execution["run_id"])
    provider_id = str(execution["provider_id"])
    model = execution.get("model") if isinstance(execution.get("model"), str) else None
    project_name = str(execution.get("project_name") or GLOBAL_TASK_PROJECT_NAME)
    project_path = str(execution.get("project_path") or _global_task_path())
    launch_message = str(execution.get("launch_message") or "")

    steps = store.list_task_steps(task_id)
    if _has_workflow_backed_steps(steps):
        await _execute_workflow_orchestration(
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            model=model,
            project_name=project_name,
            project_path=project_path,
            launch_message=launch_message,
            steps=steps,
        )
        return

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
    session_scope = f"task:{task_id}"
    try:
        selection = ChatProviderSelection(
            provider_id=provider_id,
            provider_name=_provider_name(provider_id),
            model=model,
        )
        session = await create_chat_session(
            project_name=session_scope,
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


def _has_workflow_backed_steps(steps: list[dict[str, Any]]) -> bool:
    for step in steps:
        step_input = step.get("input")
        if isinstance(step_input, dict) and step_input.get("workflow_step_id"):
            return True
    return False


async def _execute_workflow_orchestration(
    *,
    task_id: str,
    run_id: str,
    provider_id: str,
    model: str | None,
    project_name: str,
    project_path: str,
    launch_message: str,
    steps: list[dict[str, Any]],
) -> None:
    store = get_agent_store()
    store.update_run_status(run_id, "running")
    store.update_task(task_id, {"status": "running"})
    store.append_event(
        run_id=run_id,
        event_type="task.execution.started",
        provider_id=provider_id,
        app_event={
            "task_id": task_id,
            "project_path": project_path,
            "mode": "workflow",
        },
    )

    step_index = 0
    transitions = 0
    max_transitions = max(10, len(steps) * 10)
    while step_index < len(steps):
        transitions += 1
        if transitions > max_transitions:
            _finish_workflow_execution(
                task_id=task_id,
                run_id=run_id,
                status="failed",
                error={"message": "Workflow transition limit exceeded."},
            )
            return

        step = steps[step_index]
        if step.get("status") == "completed":
            step_index += 1
            continue
        step_input = step.get("input")
        if not isinstance(step_input, dict) or not step_input.get("workflow_step_id"):
            step_index += 1
            continue
        if step.get("status") == "waiting_for_user" and _step_has_user_response(step):
            _complete_step(
                task=store.get_task(task_id) or {"id": task_id},
                step_id=step["id"],
                run_id=run_id,
                output={
                    **dict(step.get("output") or {}),
                    "message": "User response accepted; workflow step resumed.",
                },
            )
            steps[step_index] = store.get_task_step(step["id"]) or step
            next_index = _apply_workflow_success_policy(
                task_id=task_id, run_id=run_id, steps=steps, completed_index=step_index
            )
            if next_index is None:
                return
            step_index = next_index
            continue
        workflow_type = str(step_input.get("workflow_type") or "llm")
        if workflow_type == "browser_action":
            completed = await _execute_browser_action_workflow_step(
                task_id=task_id,
                run_id=run_id,
                provider_id=provider_id,
                project_name=project_name,
                project_path=project_path,
                step=step,
            )
            if completed is None:
                return
            if not completed:
                next_index = _apply_workflow_failure_policy(
                    task_id=task_id,
                    run_id=run_id,
                    steps=steps,
                    failed_index=step_index,
                    error={
                        "message": f"Browser action step '{step.get('title')}' did not complete."
                    },
                )
                if next_index is None:
                    return
                steps = store.list_task_steps(task_id)
                step_index = next_index
                continue
            steps[step_index] = store.get_task_step(step["id"]) or step
            next_index = _apply_workflow_success_policy(
                task_id=task_id, run_id=run_id, steps=steps, completed_index=step_index
            )
            if next_index is None:
                return
            step_index = next_index
            continue
        if _is_app_action_workflow_type(workflow_type):
            completed = await _execute_app_action_workflow_step(
                task_id=task_id,
                run_id=run_id,
                provider_id=provider_id,
                project_name=project_name,
                project_path=project_path,
                step=step,
            )
            if completed is None:
                return
            if not completed:
                next_index = _apply_workflow_failure_policy(
                    task_id=task_id,
                    run_id=run_id,
                    steps=steps,
                    failed_index=step_index,
                    error={
                        "message": f"App action step '{step.get('title')}' did not complete."
                    },
                )
                if next_index is None:
                    return
                steps = store.list_task_steps(task_id)
                step_index = next_index
                continue
            steps[step_index] = store.get_task_step(step["id"]) or step
            next_index = _apply_workflow_success_policy(
                task_id=task_id, run_id=run_id, steps=steps, completed_index=step_index
            )
            if next_index is None:
                return
            step_index = next_index
            continue
        if workflow_type == "shell":
            completed = await _execute_shell_workflow_step(
                task_id=task_id,
                run_id=run_id,
                provider_id=provider_id,
                project_path=project_path,
                step=step,
            )
            if not completed:
                next_index = _apply_workflow_failure_policy(
                    task_id=task_id,
                    run_id=run_id,
                    steps=steps,
                    failed_index=step_index,
                    error={
                        "message": f"Shell step '{step.get('title')}' did not complete."
                    },
                )
                if next_index is None:
                    return
                steps = store.list_task_steps(task_id)
                step_index = next_index
                continue
            steps[step_index] = store.get_task_step(step["id"]) or step
            next_index = _apply_workflow_success_policy(
                task_id=task_id, run_id=run_id, steps=steps, completed_index=step_index
            )
            if next_index is None:
                return
            step_index = next_index
            continue
        if workflow_type in {
            "manual_handoff",
            "mcp_tool",
            "approval_gate",
        }:
            _wait_for_user_step(
                task_id=task_id,
                run_id=run_id,
                step=step,
                reason=workflow_type,
            )
            return
        if workflow_type == "condition":
            _complete_step(
                task=store.get_task(task_id) or {"id": task_id},
                step_id=step["id"],
                run_id=run_id,
                output={"result": "condition step completed without branching"},
            )
            steps[step_index] = store.get_task_step(step["id"]) or step
            next_index = _apply_workflow_success_policy(
                task_id=task_id, run_id=run_id, steps=steps, completed_index=step_index
            )
            if next_index is None:
                return
            step_index = next_index
            continue

        completed = await _execute_llm_workflow_step(
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            model=model,
            project_name=project_name,
            project_path=project_path,
            launch_message=launch_message,
            step=step,
            previous_steps=_previous_completed_workflow_steps(
                steps,
                current_step=step,
            ),
        )
        if not completed:
            next_index = _apply_workflow_failure_policy(
                task_id=task_id,
                run_id=run_id,
                steps=steps,
                failed_index=step_index,
                error={"message": f"Workflow step '{step.get('title')}' did not complete."},
            )
            if next_index is None:
                return
            steps = store.list_task_steps(task_id)
            step_index = next_index
            continue
        steps[step_index] = store.get_task_step(step["id"]) or step
        next_index = _apply_workflow_success_policy(
            task_id=task_id, run_id=run_id, steps=steps, completed_index=step_index
        )
        if next_index is None:
            return
        step_index = next_index

    # A run that carried on past a failed step is not a clean run: reporting it
    # as completed would put a green dot on a night where a phone never ran.
    _finish_workflow_from_steps(task_id=task_id, run_id=run_id)


async def _execute_llm_workflow_step(
    *,
    task_id: str,
    run_id: str,
    provider_id: str,
    model: str | None,
    project_name: str,
    project_path: str,
    launch_message: str,
    step: dict[str, Any],
    previous_steps: list[dict[str, Any]] | None = None,
) -> bool:
    store = get_agent_store()
    step_id = str(step["id"])
    store.update_task_step(step_id, {"status": "running"})
    store.append_event(
        run_id=run_id,
        event_type="task.step.started",
        provider_id=provider_id,
        app_event={"task_id": task_id, "step_id": step_id, "mode": "workflow"},
    )
    sink = AgentTaskRunSink(run_id=run_id)
    session_scope = f"task:{task_id}"
    try:
        session = await create_chat_session(
            project_name=session_scope,
            project_path=project_path,
            selection=ChatProviderSelection(
                provider_id=provider_id,
                provider_name=_provider_name(provider_id),
                model=model,
            ),
        )
        completed = await stream_claude_turn(
            sink,
            session,
            project_name=project_name,
            user_message=_workflow_step_message(
                step,
                launch_message,
                previous_steps=previous_steps or [],
            ),
        )
    except Exception as exc:
        logger.exception("workflow step failed task_id=%s run_id=%s step_id=%s", task_id, run_id, step_id)
        _fail_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            error={"message": str(exc)},
        )
        return False

    if sink.permission_required:
        _wait_for_user_step(
            task_id=task_id,
            run_id=run_id,
            step=step,
            reason="approval_required",
            prompt="Approval is required before this workflow step can continue.",
        )
        return False
    if sink.error_message or not completed:
        _fail_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            error={"message": sink.error_message or "Provider turn did not complete."},
        )
        return False

    output: dict[str, Any] = {"message": "Workflow step completed."}
    if sink.result_text:
        # Truncated: a step output row is read in a list, and the full turn is
        # still in the event log if anyone needs all of it.
        output["result"] = _truncate_workflow_evidence(sink.result_text, 4000)
    _complete_step(
        task=store.get_task(task_id) or {"id": task_id},
        step_id=step_id,
        run_id=run_id,
        output=output,
    )
    return True


async def _execute_shell_workflow_step(
    *,
    task_id: str,
    run_id: str,
    provider_id: str,
    project_path: str,
    step: dict[str, Any],
) -> bool:
    """Run the registered script this step names.

    No model, no tokens, no approval — the vetting happened when the script was
    registered. On failure the exit code and output stay on the step, so an
    ``on_failure: goto_step`` escalation hands an LLM step something concrete
    to diagnose instead of "the script failed".
    """
    from agent.script_store import get_script_store
    from agent.shell_step_executor import run_registered_script

    store = get_agent_store()
    step_id = str(step["id"])
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    script_id = step_input.get("script_id")

    store.update_task_step(step_id, {"status": "running"})

    script = get_script_store().get(str(script_id)) if script_id else None
    if script is None:
        error = {
            "message": f"registered script not found: {script_id}",
            "type": "ScriptNotFound",
        }
        _fail_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            error={"shell": {"status": "failed", "error": error}},
        )
        store.append_event(
            run_id=run_id,
            event_type="task.step.shell.failed",
            provider_id=provider_id,
            app_event={"task_id": task_id, "step_id": step_id, "error": error},
        )
        return False

    store.append_event(
        run_id=run_id,
        event_type="task.step.shell.started",
        provider_id=provider_id,
        app_event={
            "task_id": task_id,
            "step_id": step_id,
            "workflow_step_id": step_input.get("workflow_step_id"),
            "script_id": script["id"],
            "script_name": script["name"],
        },
    )

    result = await run_registered_script(
        script,
        extra_args=[str(item) for item in (step_input.get("script_args") or [])],
        cwd=project_path or None,
    )
    output = {"shell": result.to_output()}

    if result.completed:
        _complete_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            output=output,
        )
        store.append_event(
            run_id=run_id,
            event_type="task.step.shell.completed",
            provider_id=provider_id,
            app_event={"task_id": task_id, "step_id": step_id, "output": output},
        )
        return True

    _fail_step(
        task=store.get_task(task_id) or {"id": task_id},
        step_id=step_id,
        run_id=run_id,
        error=output,
    )
    store.append_event(
        run_id=run_id,
        event_type="task.step.shell.failed",
        provider_id=provider_id,
        app_event={"task_id": task_id, "step_id": step_id, "output": output},
    )
    return False


async def _execute_app_action_workflow_step(
    *,
    task_id: str,
    run_id: str,
    provider_id: str,
    project_name: str,
    project_path: str,
    step: dict[str, Any],
) -> bool | None:
    store = get_agent_store()
    step_id = str(step["id"])
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    actions_raw = step_input.get("actions")
    actions = (
        [action for action in actions_raw if isinstance(action, dict)]
        if isinstance(actions_raw, list)
        else []
    )

    store.update_task_step(step_id, {"status": "running"})
    store.append_event(
        run_id=run_id,
        event_type="task.step.app_action.started",
        provider_id=provider_id,
        app_event={
            "task_id": task_id,
            "step_id": step_id,
            "workflow_step_id": step_input.get("workflow_step_id"),
            "action_count": len(actions),
        },
    )

    try:
        result = await execute_app_actions(
            actions,
            context={
                "task_id": task_id,
                "run_id": run_id,
                "step_id": step_id,
                "workflow_step_id": step_input.get("workflow_step_id"),
                "project_name": project_name,
                "project_path": project_path,
                "android_device_id": step_input.get("android_device_id")
                or step_input.get("device_id"),
            },
        )
    except Exception as exc:
        logger.exception(
            "app action step failed task_id=%s run_id=%s step_id=%s",
            task_id,
            run_id,
            step_id,
        )
        error = {
            "message": str(exc) or exc.__class__.__name__,
            "type": exc.__class__.__name__,
        }
        _fail_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            error={"app_action": {"status": "failed", "error": error}},
        )
        store.append_event(
            run_id=run_id,
            event_type="task.step.app_action.failed",
            provider_id=provider_id,
            app_event={"task_id": task_id, "step_id": step_id, "error": error},
        )
        return False
    output = {"app_action": result.to_output()}
    artifact_ids = _store_app_action_artifacts(
        run_id=run_id,
        step_id=step_id,
        result=result,
    )
    if artifact_ids:
        output["artifact_ids"] = artifact_ids
        output["artifact_id"] = artifact_ids[0]

    if result.completed:
        _complete_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            output=output,
        )
        store.append_event(
            run_id=run_id,
            event_type="task.step.app_action.completed",
            provider_id=provider_id,
            app_event={"task_id": task_id, "step_id": step_id, "output": output},
        )
        return True

    if result.waiting_for_user:
        _wait_for_user_step(
            task_id=task_id,
            run_id=run_id,
            step={**step, "output": output},
            reason=result.wait_reason or "app_action",
            prompt=result.prompt or result.message or None,
        )
        store.append_event(
            run_id=run_id,
            event_type="task.step.app_action.waiting_for_user",
            provider_id=provider_id,
            app_event={"task_id": task_id, "step_id": step_id, "output": output},
        )
        return None

    error = result.error or {"message": result.message or "App action failed."}
    _fail_step(
        task=store.get_task(task_id) or {"id": task_id},
        step_id=step_id,
        run_id=run_id,
        error={**output, "error": error},
    )
    store.append_event(
        run_id=run_id,
        event_type="task.step.app_action.failed",
        provider_id=provider_id,
        app_event={"task_id": task_id, "step_id": step_id, "error": error},
    )
    return False


async def _execute_browser_action_workflow_step(
    *,
    task_id: str,
    run_id: str,
    provider_id: str,
    project_name: str,
    project_path: str,
    step: dict[str, Any],
) -> bool | None:
    store = get_agent_store()
    step_id = str(step["id"])
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    actions_raw = step_input.get("actions")
    actions = (
        [action for action in actions_raw if isinstance(action, dict)]
        if isinstance(actions_raw, list)
        else []
    )
    browser_session = _prepare_browser_session_for_execution(
        task_id=task_id,
        run_id=run_id,
        step_id=step_id,
        workflow_step_id=step_input.get("workflow_step_id"),
        enabled=bool(actions),
    )
    previous_session = _previous_browser_session_for_execution(
        run_id=run_id,
        current_session_id=browser_session.get("id") if browser_session else None,
    )

    store.update_task_step(step_id, {"status": "running"})
    store.append_event(
        run_id=run_id,
        event_type="task.step.browser_action.started",
        provider_id=provider_id,
        app_event={
            "task_id": task_id,
            "step_id": step_id,
            "workflow_step_id": step_input.get("workflow_step_id"),
            "action_count": len(actions),
        },
    )

    result = await execute_browser_actions(
        actions,
        context={
            "task_id": task_id,
            "run_id": run_id,
            "step_id": step_id,
            "workflow_step_id": step_input.get("workflow_step_id"),
            "project_name": project_name,
            "project_path": project_path,
            **_browser_session_context(
                browser_session=browser_session,
                previous_session=previous_session,
            ),
        },
    )
    output = {"browser_action": result.to_output()}
    if browser_session is not None:
        browser_session = _update_browser_session_after_action(
            browser_session=browser_session,
            result=result,
            status="resumed" if result.completed else None,
        )
        output["browser_session_id"] = browser_session["id"]
        output["browser_session"] = _browser_session_output(browser_session)
    artifact_ids = _store_browser_action_artifacts(
        run_id=run_id,
        step_id=step_id,
        result=result,
    )
    if artifact_ids:
        output["artifact_ids"] = artifact_ids
        output["artifact_id"] = artifact_ids[0]

    if result.completed:
        if browser_session is not None:
            await get_browser_runtime_manager().close_session(str(browser_session["id"]))
        _complete_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            output=output,
        )
        store.append_event(
            run_id=run_id,
            event_type="task.step.browser_action.completed",
            provider_id=provider_id,
            app_event={"task_id": task_id, "step_id": step_id, "output": output},
        )
        return True

    if result.waiting_for_user:
        handoff_session = _create_browser_handoff_session(
            task_id=task_id,
            run_id=run_id,
            step_id=step_id,
            workflow_step_id=step_input.get("workflow_step_id"),
            result=result,
        )
        checkpoint_extra: dict[str, Any] = {}
        if handoff_session is not None:
            output["browser_session_id"] = handoff_session["id"]
            output["browser_session"] = _browser_session_output(handoff_session)
            checkpoint_extra["browser_session_id"] = handoff_session["id"]
        elif browser_session is not None:
            get_browser_session_store().close(browser_session["id"])
            output.pop("browser_session_id", None)
            output.pop("browser_session", None)
        _wait_for_user_step(
            task_id=task_id,
            run_id=run_id,
            step={**step, "output": output},
            reason=result.wait_reason or "browser_action",
            prompt=result.prompt or result.message or None,
            checkpoint_extra=checkpoint_extra,
        )
        store.append_event(
            run_id=run_id,
            event_type="task.step.browser_action.waiting_for_user",
            provider_id=provider_id,
            app_event={"task_id": task_id, "step_id": step_id, "output": output},
        )
        return None

    error = result.error or {"message": result.message or "Browser action failed."}
    if browser_session is not None:
        await get_browser_runtime_manager().close_session(str(browser_session["id"]))
        get_browser_session_store().close(browser_session["id"])
    _fail_step(
        task=store.get_task(task_id) or {"id": task_id},
        step_id=step_id,
        run_id=run_id,
        error={**output, "error": error},
    )
    store.append_event(
        run_id=run_id,
        event_type="task.step.browser_action.failed",
        provider_id=provider_id,
        app_event={"task_id": task_id, "step_id": step_id, "error": error},
    )
    return False


def _prepare_browser_session_for_execution(
    *,
    task_id: str,
    run_id: str,
    step_id: str,
    workflow_step_id: Any,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    store = get_browser_session_store()
    existing = store.find_active_for_step(run_id=run_id, step_id=step_id)
    if existing is not None:
        return existing
    previous = store.latest_resumable_for_run(run_id)
    metadata = {"source": "browser_action", "live_context": True}
    if previous is not None:
        metadata["previous_browser_session_id"] = previous["id"]
        metadata["input_storage_state_path"] = previous.get("storage_state_path")
    return store.create(
        run_id=run_id,
        task_id=task_id,
        step_id=step_id,
        workflow_step_id=workflow_step_id if isinstance(workflow_step_id, str) else None,
        status="created",
        metadata=metadata,
    )


def _previous_browser_session_for_execution(
    *,
    run_id: str,
    current_session_id: str | None,
) -> dict[str, Any] | None:
    previous = get_browser_session_store().latest_resumable_for_run(run_id)
    if previous is None:
        return None
    if current_session_id and previous.get("id") == current_session_id:
        return None
    return previous


def _browser_session_context(
    *,
    browser_session: dict[str, Any] | None,
    previous_session: dict[str, Any] | None,
) -> dict[str, Any]:
    if browser_session is None:
        return {}
    storage_state_path = str(Path(str(browser_session["context_dir"])) / "storage_state.json")
    context: dict[str, Any] = {
        "browser_session_id": browser_session["id"],
        "browser_context_dir": browser_session.get("context_dir"),
        "browser_storage_state_path": storage_state_path,
    }
    previous_storage = previous_session.get("storage_state_path") if previous_session else None
    if isinstance(previous_storage, str) and previous_storage:
        context["browser_input_storage_state_path"] = previous_storage
        context["previous_browser_session_id"] = previous_session["id"]
    return context


def _update_browser_session_after_action(
    *,
    browser_session: dict[str, Any],
    result: BrowserActionAdapterResult,
    status: str | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if status:
        updates["status"] = status
    if result.storage_state_path:
        updates["storage_state_path"] = result.storage_state_path
    observations = result.observations or []
    last_observation = observations[-1] if observations else {}
    if isinstance(last_observation, dict):
        if isinstance(last_observation.get("url"), str):
            updates["current_url"] = last_observation["url"]
        if isinstance(last_observation.get("title"), str):
            updates["title"] = last_observation["title"]
    if not updates:
        return browser_session
    return get_browser_session_store().update(browser_session["id"], updates) or browser_session


def _browser_session_output(browser_session: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": browser_session["id"],
        "status": browser_session["status"],
        "expires_at": browser_session.get("expires_at"),
        "storage_state_path": browser_session.get("storage_state_path"),
    }


def _create_browser_handoff_session(
    *,
    task_id: str,
    run_id: str,
    step_id: str,
    workflow_step_id: Any,
    result: BrowserActionAdapterResult,
) -> dict[str, Any] | None:
    if result.wait_reason in {"browser_adapter_unavailable", "browser_actions_missing"}:
        return None
    observations = result.observations or []
    last_observation = observations[-1] if observations else {}
    current_url = (
        last_observation.get("url") if isinstance(last_observation, dict) else None
    )
    title = last_observation.get("title") if isinstance(last_observation, dict) else None
    store = get_browser_session_store()
    existing = store.find_active_for_step(run_id=run_id, step_id=step_id)
    if existing is None:
        session = store.create(
            run_id=run_id,
            task_id=task_id,
            step_id=step_id,
            workflow_step_id=workflow_step_id if isinstance(workflow_step_id, str) else None,
            status="waiting_for_user",
            current_url=current_url if isinstance(current_url, str) else None,
            title=title if isinstance(title, str) else None,
            handoff_reason=result.wait_reason,
            metadata={
                "source": "browser_action",
                "live_context": False,
                "message": result.message,
            },
        )
    else:
        session = store.mark_waiting(
            existing["id"],
            reason=result.wait_reason,
            current_url=current_url if isinstance(current_url, str) else None,
            title=title if isinstance(title, str) else None,
        )
    return session


def _store_browser_action_artifacts(
    *,
    run_id: str,
    step_id: str,
    result: BrowserActionAdapterResult,
) -> list[str]:
    artifact_ids: list[str] = []
    store = get_agent_store()
    for index, screenshot in enumerate(result.screenshots, start=1):
        path = Path(str(screenshot)).expanduser()
        artifact = store.add_artifact(
            run_id=run_id,
            kind="browser_screenshot",
            path=str(path),
            mime_type="image/png",
            metadata={"step_id": step_id, "action_index": index},
        )
        artifact_id = artifact.get("id") if isinstance(artifact, dict) else None
        if isinstance(artifact_id, str):
            artifact_ids.append(artifact_id)
    return artifact_ids


def _store_app_action_artifacts(
    *,
    run_id: str,
    step_id: str,
    result: AppActionAdapterResult,
) -> list[str]:
    artifact_ids: list[str] = []
    store = get_agent_store()
    for index, screenshot in enumerate(result.screenshots, start=1):
        path = Path(str(screenshot)).expanduser()
        artifact = store.add_artifact(
            run_id=run_id,
            kind="app_screenshot",
            path=str(path),
            mime_type="image/png",
            metadata={"step_id": step_id, "action_index": index},
        )
        artifact_id = artifact.get("id") if isinstance(artifact, dict) else None
        if isinstance(artifact_id, str):
            artifact_ids.append(artifact_id)
    return artifact_ids


def _is_app_action_workflow_type(workflow_type: str) -> bool:
    return workflow_type in {
        "app_action",
        "android_action",
        "mobile_action",
        "device_action",
    }


def _apply_workflow_success_policy(
    *,
    task_id: str,
    run_id: str,
    steps: list[dict[str, Any]],
    completed_index: int,
) -> int | None:
    """Where execution goes after a step succeeds.

    Returns the next index, or ``None`` when the workflow is finished here.
    Default is the next step, which is what every workflow did before this
    existed. ``end`` is what lets a diagnosis step be reachable only through
    ``on_failure: goto_step`` instead of running on every clean night.
    """
    store = get_agent_store()
    step = steps[completed_index]
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    policy = step_input.get("on_success")
    if not isinstance(policy, dict):
        return completed_index + 1
    policy_type = str(policy.get("type") or "continue")

    if policy_type == "goto_step":
        target = policy.get("target_step_id") or policy.get("step_id") or policy.get("target")
        target_index = _workflow_step_index(steps, target) if isinstance(target, str) else None
        if target_index is None:
            _finish_workflow_execution(
                task_id=task_id,
                run_id=run_id,
                status="failed",
                error={"message": f"on_success goto_step target not found: {target}"},
            )
            return None
        store.append_event(
            run_id=run_id,
            event_type="task.step.goto",
            app_event={
                "task_id": task_id,
                "step_id": step["id"],
                "target_step_id": target,
                "reason": "on_success",
            },
        )
        return target_index

    if policy_type == "end":
        _finish_workflow_from_steps(task_id=task_id, run_id=run_id)
        return None

    return completed_index + 1


def _finish_workflow_from_steps(*, task_id: str, run_id: str) -> None:
    """Close a workflow, reporting failure if any step failed along the way.

    Shared by the natural end of the loop and by ``on_success: end`` so both
    routes tell the same truth: carrying on past a failure, or stopping early
    after one, is still a failed run.
    """
    store = get_agent_store()
    failed_steps = [
        step for step in store.list_task_steps(task_id) if step.get("status") == "failed"
    ]
    if failed_steps:
        _finish_workflow_execution(
            task_id=task_id,
            run_id=run_id,
            status="failed",
            error={
                "message": (
                    f"{len(failed_steps)} step(s) failed; the workflow continued past them."
                ),
                "failed_step_ids": [step["id"] for step in failed_steps],
            },
        )
        return
    _finish_workflow_execution(
        task_id=task_id,
        run_id=run_id,
        status="completed",
        result={"message": "Workflow completed."},
    )


def _apply_workflow_failure_policy(
    *,
    task_id: str,
    run_id: str,
    steps: list[dict[str, Any]],
    failed_index: int,
    error: dict[str, Any],
) -> int | None:
    store = get_agent_store()
    failed_step = steps[failed_index]
    step_input = failed_step.get("input") if isinstance(failed_step.get("input"), dict) else {}
    policy = step_input.get("on_failure")
    if not isinstance(policy, dict):
        policy = {"type": "abort"}
    policy_type = str(policy.get("type") or "abort")

    if policy_type == "retry":
        retry_state = step_input.get("retry_state")
        if not isinstance(retry_state, dict):
            retry_state = {}
        attempts = int(retry_state.get("attempts") or 0)
        max_attempts = int(policy.get("max_attempts") or policy.get("max_retries") or 1)
        if attempts < max_attempts:
            retry_state = {**retry_state, "attempts": attempts + 1}
            next_input = {**step_input, "retry_state": retry_state}
            output = dict(failed_step.get("output") or {})
            output["last_retry_error"] = error
            store.update_task_step(
                failed_step["id"],
                {
                    "status": "queued",
                    "input": next_input,
                    "output": output,
                },
            )
            store.append_event(
                run_id=run_id,
                event_type="task.step.retry_scheduled",
                app_event={
                    "task_id": task_id,
                    "step_id": failed_step["id"],
                    "attempt": attempts + 1,
                    "max_attempts": max_attempts,
                },
            )
            return failed_index
        then_policy = policy.get("then")
        if isinstance(then_policy, dict):
            return _apply_terminal_or_branch_policy(
                task_id=task_id,
                run_id=run_id,
                steps=steps,
                failed_index=failed_index,
                failed_step=failed_step,
                policy=then_policy,
                error=error,
            )
        _finish_workflow_execution(
            task_id=task_id,
            run_id=run_id,
            status="failed",
            error=error,
        )
        return None

    return _apply_terminal_or_branch_policy(
        task_id=task_id,
        run_id=run_id,
        steps=steps,
        failed_index=failed_index,
        failed_step=failed_step,
        policy=policy,
        error=error,
    )


def _apply_terminal_or_branch_policy(
    *,
    task_id: str,
    run_id: str,
    steps: list[dict[str, Any]],
    failed_index: int,
    failed_step: dict[str, Any],
    policy: dict[str, Any],
    error: dict[str, Any],
) -> int | None:
    policy_type = str(policy.get("type") or "abort")
    if policy_type == "goto":
        policy_type = "goto_step"

    if policy_type == "continue":
        # The step stays failed — this is not a pass. The run carries on so
        # that work which does not depend on it still happens, and the failure
        # is visible on the step and in the timeline.
        get_agent_store().append_event(
            run_id=run_id,
            event_type="task.step.continued_after_failure",
            app_event={
                "task_id": task_id,
                "step_id": failed_step["id"],
                "workflow_step_id": (failed_step.get("input") or {}).get("workflow_step_id"),
                "error": error,
            },
        )
        return failed_index + 1

    if policy_type == "goto_step":
        target = policy.get("target_step_id") or policy.get("step_id") or policy.get("target")
        if not isinstance(target, str) or not target:
            _finish_workflow_execution(
                task_id=task_id,
                run_id=run_id,
                status="failed",
                error={"message": "goto_step failure policy is missing target_step_id."},
            )
            return None
        target_index = _workflow_step_index(steps, target)
        if target_index is None:
            _finish_workflow_execution(
                task_id=task_id,
                run_id=run_id,
                status="failed",
                error={"message": f"goto_step target not found: {target}"},
            )
            return None
        get_agent_store().append_event(
            run_id=run_id,
            event_type="task.step.goto",
            app_event={
                "task_id": task_id,
                "step_id": failed_step["id"],
                "target_step_id": target,
            },
        )
        return target_index

    if policy_type in {"ask_user", "manual_handoff"}:
        _wait_for_user_step(
            task_id=task_id,
            run_id=run_id,
            step=failed_step,
            reason=policy_type,
            prompt=policy.get("prompt") if isinstance(policy.get("prompt"), str) else None,
        )
        return None

    _finish_workflow_execution(
        task_id=task_id,
        run_id=run_id,
        status="failed",
        error=error,
    )
    return None


def _workflow_step_index(steps: list[dict[str, Any]], workflow_step_id: str) -> int | None:
    for index, step in enumerate(steps):
        step_input = step.get("input")
        if not isinstance(step_input, dict):
            continue
        if step_input.get("workflow_step_id") == workflow_step_id:
            return index
    return None


def _wait_for_user_step(
    *,
    task_id: str,
    run_id: str,
    step: dict[str, Any],
    reason: str,
    prompt: str | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
) -> None:
    store = get_agent_store()
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    on_failure = step_input.get("on_failure")
    if not isinstance(on_failure, dict):
        on_failure = {}
    checkpoint = {
        "status": "waiting_for_user",
        "reason": reason,
        "prompt": prompt or _checkpoint_prompt(step, on_failure),
        "workflow_step_id": step_input.get("workflow_step_id"),
        "step_id": step.get("id"),
        "step_title": step.get("title"),
        "workflow_type": step_input.get("workflow_type"),
        "success_criteria": step_input.get("success_criteria"),
        "resume": on_failure.get("resume") or "same_step",
        "resume_step_id": on_failure.get("resume_step_id"),
        "resume_behavior": _checkpoint_resume_behavior(on_failure),
        "resume_label": _checkpoint_resume_label(on_failure),
        "required_user_action": _required_user_action(reason),
        "allow_memory": _checkpoint_allows_memory(reason, step_input),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if checkpoint_extra:
        checkpoint.update(checkpoint_extra)
    output = dict(step.get("output") or {})
    output["checkpoint"] = checkpoint
    output["reason"] = reason
    updated_step = store.update_task_step(
        step["id"],
        {"status": "waiting_for_user", "output": output},
    )
    task = store.get_task(task_id)
    metadata = dict(task.get("metadata") or {}) if task else {}
    metadata["active_checkpoint"] = {
        "run_id": run_id,
        "step_id": step["id"],
        "workflow_step_id": checkpoint.get("workflow_step_id"),
        "reason": reason,
    }
    store.update_task(task_id, {"status": "waiting_for_user", "metadata": metadata})
    store.update_run_status(run_id, "waiting_for_user")
    store.append_event(
        run_id=run_id,
        event_type="task.step.waiting_for_user",
        app_event={
            "task_id": task_id,
            "step_id": step["id"],
            "checkpoint": checkpoint,
            "step": updated_step,
        },
    )


def _finish_workflow_execution(
    *,
    task_id: str,
    run_id: str,
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
    task = store.get_task(task_id)
    metadata = dict(task.get("metadata") or {}) if task else {}
    metadata.pop("active_checkpoint", None)
    updates["metadata"] = metadata
    store.update_task(task_id, updates)
    store.append_event(
        run_id=run_id,
        event_type=f"task.execution.{status}",
        app_event={"task_id": task_id, "result": result or {}, "error": error or {}},
    )


def _workflow_step_message(
    step: dict[str, Any],
    launch_message: str,
    *,
    previous_steps: list[dict[str, Any]] | None = None,
) -> str:
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    instruction = _workflow_text(
        step_input.get("instruction") or step_input.get("description")
    )
    observation = _workflow_text(step_input.get("observation"))
    memory_read = _workflow_jsonish(
        step_input.get("memory_read") or step_input.get("memoryRead")
    )
    memory_write = _workflow_jsonish(
        step_input.get("memory_write") or step_input.get("memoryWrite")
    )
    tool_hint = _workflow_text(step_input.get("tool_hint"))
    actions = _workflow_jsonish(step_input.get("actions")) if step_input.get("actions") else ""
    on_failure = _workflow_jsonish(step_input.get("on_failure"))
    lines = [
        launch_message,
        "",
        "Current workflow step:",
        f"- id: {step_input.get('workflow_step_id') or step.get('id')}",
        f"- title: {step.get('title')}",
        f"- type: {step_input.get('workflow_type') or 'llm'}",
    ]
    if instruction:
        lines.append(f"- instruction: {instruction}")
    if observation:
        lines.append(f"- observation: {observation}")
    if memory_read:
        lines.append(f"- memory read: {memory_read}")
    if memory_write:
        lines.append(f"- memory write: {memory_write}")
    if tool_hint:
        lines.append(f"- tool_hint: {tool_hint}")
    if actions:
        lines.append(f"- actions: {actions}")
    evidence = _workflow_previous_evidence(previous_steps or [])
    if evidence:
        lines.extend(
            [
                "",
                "Previous workflow evidence:",
                evidence,
                "",
                (
                    "Use the evidence above as the source of truth for this step. "
                    "Do not re-open URLs, call curl, run web search, or repeat browser/device "
                    "checks unless this current step explicitly lists executable actions that "
                    "require new collection."
                ),
            ]
        )
    lines.extend(
        [
            f"- success_criteria: {step_input.get('success_criteria') or 'not specified'}",
            f"- on_failure: {on_failure or 'not specified'}",
            "",
            "Complete only this workflow step and report the result.",
        ]
    )
    return "\n".join(lines)


# Failed steps are evidence too — usually the most important kind. A workflow
# that escalates (``on_failure: goto_step: diagnose``) sends an LLM step to work
# out *why* the previous step broke; excluding the failed step left that LLM
# with nothing and it (correctly) refused to guess.
_EVIDENCE_STEP_STATUSES = {"completed", "failed"}


def _previous_completed_workflow_steps(
    steps: list[dict[str, Any]],
    *,
    current_step: dict[str, Any],
) -> list[dict[str, Any]]:
    current_sequence = current_step.get("sequence")
    current_id = current_step.get("id")
    previous: list[dict[str, Any]] = []
    for step in steps:
        if step.get("status") not in _EVIDENCE_STEP_STATUSES:
            continue
        if current_sequence is not None and step.get("sequence") is not None:
            try:
                if int(step["sequence"]) >= int(current_sequence):
                    continue
            except (TypeError, ValueError):
                pass
        elif current_id is not None and step.get("id") == current_id:
            break
        previous.append(step)
    return previous[-5:]


def _workflow_previous_evidence(steps: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, step in enumerate(steps, start=1):
        output = step.get("output")
        if not isinstance(output, dict) or not output:
            continue
        step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
        lines = [
            f"{index}. {step.get('title') or step_input.get('workflow_step_id') or step.get('id')}",
            f"   workflow_step_id: {step_input.get('workflow_step_id') or step.get('id')}",
        ]
        summary = _workflow_step_output_summary(output)
        if summary:
            lines.extend(f"   {line}" for line in summary)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _workflow_step_output_summary(output: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    shell = output.get("shell")
    if isinstance(shell, dict):
        # The whole point of a shell step's evidence is the escalation case:
        # the next step is often an LLM asked to work out *why* this failed, so
        # give it the exit code and the tail of both streams rather than a
        # status word.
        lines.append(f"shell.status: {shell.get('status')}")
        lines.append(f"shell.exit_code: {shell.get('exit_code')}")
        if shell.get("timed_out"):
            lines.append("shell.timed_out: true")
        command = shell.get("command")
        if isinstance(command, list) and command:
            lines.append("shell.command: " + " ".join(str(part) for part in command))
        for stream in ("stdout", "stderr"):
            text = shell.get(stream)
            if isinstance(text, str) and text.strip():
                lines.append(
                    f"shell.{stream}: "
                    + _truncate_workflow_evidence(text.strip(), 2000)
                )
        error = shell.get("error")
        if isinstance(error, str) and error:
            lines.append(f"shell.error: {error}")
        elif isinstance(error, dict) and error.get("message"):
            lines.append(f"shell.error: {error['message']}")
    browser_action = output.get("browser_action")
    if isinstance(browser_action, dict):
        lines.append(f"browser_action.status: {browser_action.get('status')}")
        observations = browser_action.get("observations")
        if isinstance(observations, list):
            for obs in observations[-5:]:
                if not isinstance(obs, dict):
                    continue
                parts = []
                for key in ("action_index", "url", "title", "text", "label"):
                    value = obs.get(key)
                    if value is not None and value != "":
                        parts.append(f"{key}={_workflow_text(value)}")
                action = obs.get("action")
                if isinstance(action, dict):
                    action_type = action.get("type")
                    target = action.get("target") or action.get("url")
                    if action_type:
                        parts.append(f"action={action_type}")
                    if target:
                        parts.append(f"target={_workflow_text(target)}")
                if parts:
                    lines.append(f"observation: {'; '.join(parts)}")
        screenshots = browser_action.get("screenshots")
        if isinstance(screenshots, list) and screenshots:
            lines.append(
                "screenshots: "
                + ", ".join(str(path) for path in screenshots[-3:])
            )
    app_action = output.get("app_action")
    if isinstance(app_action, dict):
        lines.append(f"app_action.status: {app_action.get('status')}")
        observations = app_action.get("observations")
        if isinstance(observations, list):
            for obs in observations[-5:]:
                if isinstance(obs, dict):
                    lines.append(
                        "observation: "
                        + _truncate_workflow_evidence(
                            json.dumps(obs, ensure_ascii=False, sort_keys=True),
                            500,
                        )
                    )
        screenshots = app_action.get("screenshots")
        if isinstance(screenshots, list) and screenshots:
            lines.append(
                "screenshots: "
                + ", ".join(str(path) for path in screenshots[-3:])
            )
    artifact_ids = output.get("artifact_ids")
    if isinstance(artifact_ids, list) and artifact_ids:
        lines.append("artifact_ids: " + ", ".join(str(item) for item in artifact_ids[-5:]))
    artifact_id = output.get("artifact_id")
    if isinstance(artifact_id, str) and artifact_id:
        lines.append(f"artifact_id: {artifact_id}")
    if not lines:
        lines.append(
            "output: "
            + _truncate_workflow_evidence(
                json.dumps(output, ensure_ascii=False, sort_keys=True),
                1000,
            )
        )
    return lines


def _truncate_workflow_evidence(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _workflow_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _workflow_jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _workflow_text(value)


def _step_has_user_response(step: dict[str, Any]) -> bool:
    output = step.get("output")
    if not isinstance(output, dict):
        return False
    responses = output.get("user_responses")
    return isinstance(responses, list) and len(responses) > 0


def _resume_launch_message(task: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return str(task.get("goal") or task.get("description") or task.get("title") or "")


def _checkpoint_prompt(step: dict[str, Any], on_failure: dict[str, Any]) -> str:
    prompt = on_failure.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    title = str(step.get("title") or "this step")
    return f"User input is required before continuing workflow step: {title}."


def _checkpoint_resume_behavior(on_failure: dict[str, Any]) -> str:
    resume = str(on_failure.get("resume") or "same_step")
    if resume == "next_step":
        return "continue_next_step"
    if resume == "target_step":
        return "continue_target_step"
    # The current runtime accepts the user response as evidence that the
    # waiting step is complete, then continues the workflow. It does not rerun
    # the same adapter action automatically.
    return "complete_waiting_step_then_continue"


def _checkpoint_resume_label(on_failure: dict[str, Any]) -> str:
    behavior = _checkpoint_resume_behavior(on_failure)
    if behavior == "continue_next_step":
        return "응답을 저장하고 다음 단계로 진행합니다."
    if behavior == "continue_target_step":
        target = on_failure.get("resume_step_id")
        if isinstance(target, str) and target.strip():
            return f"응답을 저장하고 {target.strip()} 단계로 이동합니다."
        return "응답을 저장하고 지정된 단계로 이동합니다."
    return "응답을 저장하고 현재 대기 단계를 완료 처리한 뒤 다음 단계로 진행합니다."


def _checkpoint_allows_memory(reason: str, step_input: dict[str, Any]) -> bool:
    if reason != "ask_user":
        return False
    workflow_type = str(step_input.get("workflow_type") or "")
    if workflow_type in {"manual_handoff", "browser_action", "app_action", "android_action", "mobile_action", "device_action"}:
        return False
    return True


def _required_user_action(reason: str) -> str:
    if reason == "browser_action":
        return "브라우저 작업을 완료하거나 확인한 뒤 워크플로를 재개하세요."
    if reason == "mcp_tool":
        return "요청된 MCP/도구 작업을 확인하거나 완료한 뒤 워크플로를 재개하세요."
    if reason == "approval_gate":
        return "필요한 작업을 승인하거나 거절한 뒤 워크플로를 재개하세요."
    if reason == "approval_required":
        return "대기 중인 요청을 승인한 뒤 워크플로를 재개하세요."
    return "요청된 수동 작업을 완료한 뒤 워크플로를 재개하세요."


def _execution_is_dry_run(execution: dict[str, Any], *, store: Any) -> bool:
    if bool(execution.get("dry_run")):
        return True
    task = execution.get("task")
    if isinstance(task, dict) and bool(task.get("dry_run")):
        return True
    run_id = execution.get("run_id")
    if isinstance(run_id, str) and run_id:
        run = store.get_run(run_id)
        return bool(run and run.get("dry_run"))
    return False


async def _simulate_run(execution: dict[str, Any]) -> None:
    """Emit dry-run timeline events without invoking providers, tools, or policy."""
    store = get_agent_store()
    run_id = str(execution.get("run_id") or "")
    if not run_id:
        return

    run = store.get_run(run_id)
    if run is None:
        return

    task = execution.get("task")
    if not isinstance(task, dict):
        task_id = execution.get("task_id") or run.get("task_id")
        task = store.get_task(str(task_id)) if task_id else None

    agent = execution.get("agent")
    if not isinstance(agent, dict):
        agent_id = execution.get("agent_id") or run.get("agent_id")
        agent = store.get_agent(str(agent_id)) if agent_id else None
    if not isinstance(agent, dict):
        agent = {
            "id": run.get("agent_id"),
            "name": "Dry run preview",
            "tools_json": [],
            "flow_json": execution.get("workflow") if isinstance(execution.get("workflow"), list) else [],
        }

    store.update_run_status(run_id, "running")
    store.append_event(
        run_id=run_id,
        event_type="run.started",
        app_event={
            "dry_run": True,
            "agent_id": agent.get("id"),
            "task_id": task.get("id") if isinstance(task, dict) else None,
        },
    )
    summary = store.simulate_run_from_workflow(
        agent=agent,
        task=task if isinstance(task, dict) else None,
        run_id=run_id,
    )
    store.append_event(
        run_id=run_id,
        event_type="run.complete",
        app_event={
            "dry_run": True,
            "summary": summary["summary"],
            "steps_simulated": summary["steps_simulated"],
        },
    )
    store.update_run_status(run_id, "completed")


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
    workflow_steps: list[dict[str, Any]] | None = None,
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
    for workflow_step in workflow_steps or []:
        _select_workflow_step_capabilities(workflow_step, add_by)
    return selected


def _select_workflow_step_capabilities(workflow_step: dict[str, Any], add_by: Any) -> None:
    step_type = str(workflow_step.get("type") or "").strip()
    tool_hint = str(workflow_step.get("tool_hint") or "").strip()
    if step_type == "browser_action":
        add_by(lambda cap: cap.get("type") == "mcp_server" and cap.get("name") == "browser")
    if _is_app_action_workflow_type(step_type):
        add_by(lambda cap: cap.get("type") == "builtin" and cap.get("name") == "device.control")
    if tool_hint:
        add_by(
            lambda cap, tool_hint=tool_hint: tool_hint
            in {
                str(cap.get("id") or ""),
                str(cap.get("name") or ""),
                str(cap.get("source") or ""),
            }
        )


def _plan_steps(
    task: dict[str, Any],
    capabilities: list[dict[str, Any]],
    *,
    workflow_steps: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    workflow_steps = workflow_steps if workflow_steps is not None else _workflow_steps_for_task(task)
    if workflow_steps:
        return _plan_workflow_steps(workflow_steps, capabilities)

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


def _workflow_steps_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    agent_id = task.get("assigned_agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        return []
    agent = get_agent_store().get_agent(agent_id)
    if not agent or agent.get("is_pseudo"):
        return []
    flow_json = agent.get("flow_json")
    if not flow_json:
        return []
    try:
        return normalize_workflow(flow_json)
    except WorkflowNormalizationError as exc:
        logger.warning(
            "Ignoring invalid workflow for task %s agent %s: %s",
            task.get("id"),
            agent_id,
            exc,
        )
        return []


def _plan_workflow_steps(
    workflow_steps: list[dict[str, Any]],
    capabilities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for index, workflow_step in enumerate(workflow_steps, start=1):
        planned.append(
            {
                "title": _workflow_step_title(workflow_step, index),
                "capability_id": _workflow_step_capability_id(workflow_step, capabilities),
                "status": "queued",
                "input": {
                    "workflow_step_id": workflow_step["id"],
                    "workflow_type": workflow_step["type"],
                    "description": workflow_step.get("description") or "",
                    "instruction": workflow_step.get("instruction")
                    or workflow_step.get("description")
                    or "",
                    "observation": workflow_step.get("observation") or "",
                    "memory_read": workflow_step.get("memory_read")
                    or workflow_step.get("memoryRead")
                    or "",
                    "memory_write": workflow_step.get("memory_write")
                    or workflow_step.get("memoryWrite")
                    or "",
                    "tool_hint": workflow_step.get("tool_hint"),
                    "device_id": workflow_step.get("device_id"),
                    "android_device_id": workflow_step.get("android_device_id"),
                    "actions": workflow_step.get("actions") or [],
                    "script_id": workflow_step.get("script_id"),
                    "script_args": workflow_step.get("script_args") or [],
                    "success_criteria": workflow_step.get("success_criteria") or "",
                    "on_failure": workflow_step.get("on_failure") or {"type": "abort"},
                    "on_success": workflow_step.get("on_success") or {"type": "continue"},
                    "retry_state": {"attempts": 0},
                },
            }
        )
    return planned


def _workflow_step_title(step: dict[str, Any], index: int) -> str:
    name = str(step.get("name") or "").strip()
    if name:
        return name
    return f"Workflow step {index}"


def _workflow_step_capability_id(
    workflow_step: dict[str, Any],
    capabilities: list[dict[str, Any]],
) -> str | None:
    tool_hint = str(workflow_step.get("tool_hint") or "").strip()
    if tool_hint:
        for capability in capabilities:
            if tool_hint in {str(capability.get("id") or ""), str(capability.get("name") or "")}:
                return capability.get("id")
    if workflow_step.get("type") == "llm":
        llm_cap = _first_capability(capabilities, "llm_cli")
        return llm_cap.get("id") if llm_cap else None
    return None


def _build_launch_message(
    task: dict[str, Any],
    *,
    capabilities: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    user_goal: str,
    system_prompt: str,
) -> str:
    capability_lines = [
        f"- {capability.get('type')}:{capability.get('name')} status={capability.get('status')}"
        for capability in capabilities
    ]
    step_lines = [f"{index + 1}. {step.get('title')}" for index, step in enumerate(steps)]
    return "\n".join(
        [
            system_prompt,
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


def _task_goal_text(task: dict[str, Any], override_prompt: str | None) -> str:
    if override_prompt and override_prompt.strip():
        return override_prompt.strip()
    return str(task.get("goal") or task.get("description") or task.get("title") or "").strip()


def _compose_assigned_agent_prompt(task: dict[str, Any], *, task_goal: str) -> str | None:
    agent_id = task.get("assigned_agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        return None
    store = get_agent_store()
    agent = store.get_agent(agent_id)
    if not agent or agent.get("is_pseudo"):
        return None
    memories = store.list_memories(agent_id, limit=100) or []
    return compose_system_prompt({**agent, "memories": memories}, task_goal=task_goal)


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
