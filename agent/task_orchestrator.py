"""Task orchestration service for Work Cockpit."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from approvals.approval_service import expire_approval
from approvals.approval_store import get_approval_store, is_request_expired
from audit.route_audit import record_api_action
from chat.chat_session_service import ChatProviderSelection, create_chat_session, get_chat_provider_selection
from chat.chat_stream_service import stream_claude_turn
from core.config import get_config
from core.runtime_paths import runtime_dir
from llm.claude_session import get_session_manager
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
from .capability_registry import (
    BROWSER_RUNTIME_CAPABILITY_NAME,
    detected_mcp_server_configs,
    refresh_capability_registry,
    verify_declared_mcp_ids,
)
from .configurator import is_builder_added_tool
from .prompt_composer import compose_system_prompt
from .cli_agent_runtime import find_cli_agent_source_path, resolve_cli_agent_definition
from .cli_agent_sources import cli_agent_reference_prompt
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
        # What the turn parked on, when it parked on an approval: the approval
        # row the user will decide, the tool that asked, and the concrete thing
        # it wants to touch. Without this the checkpoint can only say "approval
        # required", and the phone shows a card with nothing on it that says
        # which run or which file it belongs to.
        self.permission_denial: dict[str, Any] | None = None
        # Tool calls that were *refused* during this turn, as opposed to parked
        # on. A policy denial (`chat_stream_service` "permission.policy_denied")
        # never stops the turn: the SDK is answered with a deny result and the
        # model carries on, so the turn still ends `subtype: success` and
        # nothing downstream could tell that the work was refused. Recording it
        # here is what lets the step be judged on what happened inside it.
        self.denied_tool_calls: list[dict[str, Any]] = []
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
            self.permission_denial = self._extract_permission_denial(data)
        if event_type == "app_event" and data.get("event") == "permission.policy_denied":
            self.denied_tool_calls.append(self._extract_policy_denial(data))
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
    def _extract_permission_denial(data: dict[str, Any]) -> dict[str, Any] | None:
        """Pull the approval identity out of a ``permission_required`` event.

        ``chat_stream_service`` sends one denial per parked tool call
        (``denials[0]``) and repeats the approval id at the top level. Read the
        denial first and fall back to the envelope, so an older or partial
        payload still yields whatever identity it does carry rather than
        nothing at all.
        """
        denials = data.get("denials")
        first = (
            denials[0]
            if isinstance(denials, list) and denials and isinstance(denials[0], dict)
            else {}
        )
        approval_id = first.get("approval_id") or data.get("approval_id")
        tool_name = first.get("tool_name")
        tool_input = first.get("input")
        denial: dict[str, Any] = {
            "approval_id": approval_id if isinstance(approval_id, str) else None,
            "tool_name": tool_name if isinstance(tool_name, str) else None,
            "input": tool_input if isinstance(tool_input, dict) else {},
        }
        return denial

    @staticmethod
    def _extract_policy_denial(data: dict[str, Any]) -> dict[str, Any]:
        """Pull the refused tool out of a ``permission.policy_denied`` event.

        The app event carries the tool name in ``detail`` and the standing
        rule's own words in ``data.reason``; both are optional, and a denial
        with neither is still a denial, so nothing here is allowed to make the
        record disappear.
        """
        payload = data.get("data") if isinstance(data.get("data"), dict) else {}
        tool_name = data.get("detail")
        reason = payload.get("reason")
        return {
            "source": "policy",
            "tool_name": tool_name if isinstance(tool_name, str) else None,
            "reason": reason if isinstance(reason, str) else None,
        }

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


def resume_task_orchestration(
    task_id: str,
    *,
    permission_decision: str | None = None,
) -> dict[str, Any] | None:
    """Build execution payload for resuming a task from its active checkpoint.

    ``permission_decision`` is what the user just pressed on an approval card
    ("approve_once", "deny", …). It rides along in the execution payload so the
    parked llm step can answer the provider's still-open permission callback
    with it instead of guessing, or re-reading a row that may not have landed
    yet. Left unset (a plain "resume" tap) the step reads the approval record
    itself and stays parked if nothing has been decided.
    """
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
    execution: dict[str, Any] = {
        "auto_start": True,
        "resume": True,
        "run_id": run_id,
        "task_id": task_id,
        "project_name": project_name,
        "project_path": project_path,
        "provider_id": provider_id,
        "model": run.get("model"),
        "launch_message": launch_message,
    }
    if isinstance(permission_decision, str) and permission_decision.strip():
        execution["permission_decision"] = permission_decision.strip()
    return {
        "task": task,
        "run": run,
        "checkpoint": checkpoint,
        "execution": execution,
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

    if workflow_type == "notify":
        completed = await _execute_notify_workflow_step(
            task_id=task_id, run_id=run_id, step=step
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
            # This run's steps only: evidence handed to an LLM step must come
            # from the run it is part of, not from whatever the same task did
            # on some earlier night.
            _steps_for_run(store, task_id, run_id),
            current_step=step,
        ),
    )
    if completed is None:
        # Parked on an approval, same as the browser/app branches above.
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
    permission_decision_raw = execution.get("permission_decision")
    permission_decision = (
        permission_decision_raw if isinstance(permission_decision_raw, str) else None
    )

    steps = _steps_for_run(store, task_id, run_id)
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
            permission_decision=permission_decision,
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
        # The workflow-less path runs the agent's tools too, so it gets the
        # same injection and the same report. Leaving it out would mean an
        # agent's declared servers reached the runtime only when it happened
        # to have a workflow.
        await _apply_declared_mcp_servers(
            session,
            agent_id=_step_agent_id(store.get_task(task_id) or {}),
            run_id=run_id,
            task_id=task_id,
            provider_id=provider_id,
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
        # Same identity *and the same words* the workflow path records on its
        # checkpoint. The run status stays `blocked` (that is what a non-
        # workflow run's step and every reader of it already call this), but
        # the reason, the sentence, and the required user action are the ones
        # the workflow path writes, so the two paths stop describing one
        # situation two ways.
        denial = sink.permission_denial or {}
        tool_name = denial.get("tool_name")
        tool_target = _approval_tool_target(denial.get("input"))
        _finish_execution(
            task_id=task_id,
            run_id=run_id,
            execute_step=execute_step,
            summary_step=summary_step,
            status="blocked",
            error={
                "message": _approval_wait_prompt(tool_name, tool_target),
                "reason": APPROVAL_WAIT_REASON,
                "required_user_action": _required_user_action(APPROVAL_WAIT_REASON),
                "approval_id": denial.get("approval_id"),
                "tool_name": tool_name,
                "tool_target": tool_target,
            },
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
    if sink.denied_tool_calls:
        # Same rule as the workflow path (`_DENIED_STEP_MESSAGE`): a standing
        # rule refused a tool, the turn carried on and ended clean, and calling
        # that "Provider turn completed." would report a refusal as a success.
        # This path has no `on_failure`, so the run simply ends failed.
        _finish_execution(
            task_id=task_id,
            run_id=run_id,
            execute_step=execute_step,
            summary_step=summary_step,
            status="failed",
            error=_denied_step_error(sink.denied_tool_calls, sink.result_text),
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


# What the scheduler calls "doing work" (agent/scheduler.py). A run in one of
# these has not finished, so the schedule that owns it keeps skipping.
_UNFINISHED_RUN_STATUSES = {"queued", "starting", "running"}


def _steps_for_run(store: Any, task_id: str, run_id: str) -> list[dict[str, Any]]:
    """The steps belonging to *this* run, in order.

    ``list_task_steps`` is task-scoped, and every fire of a schedule creates a
    fresh set of step rows on the same task — an agent that has run for three
    months has hundreds. Walking that whole history instead of the current
    run's own steps is how a scheduled agent ate itself on 2026-08-06: the
    loop skipped every ``completed`` row, landed on a ``shell`` step left
    ``running`` by a run three days earlier, ran it, and when it failed its
    ``on_failure: goto_step: diagnose`` resolved to the *first* ``diagnose``
    row on the task (long since completed), so the loop walked forward, hit
    the same stale shell step again, and re-ran a 52-minute device script ten
    times over 8h45m — the run never reaching a terminal state, and the
    schedule skipping every firing behind it.

    Rows written before steps carried a ``run_id`` (and hand-built rows in
    tests) have none, and those still have to run — so fall back to the full
    list, but only when *no row on the task is stamped at all*. Falling back
    whenever this particular run has no rows yet would re-open the exact hole
    above: a run whose steps are missing for any other reason would inherit
    every earlier fire's history and start replaying it. Legacy data is a
    property of the task, not of one run, so that is what we test for.
    """
    steps = store.list_task_steps(task_id)
    scoped = [step for step in steps if step.get("run_id") == run_id]
    if scoped:
        return scoped
    if any(step.get("run_id") for step in steps):
        # The task stamps its steps, so this run genuinely owns none. Running
        # someone else's is worse than running nothing.
        return []
    return steps


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
    permission_decision: str | None = None,
) -> None:
    """Drive a workflow run and make sure it ends.

    Every exit from the step loop below has to leave the run in a terminal
    state (or deliberately parked on a human). An unfinished run is not just a
    stale row: ``skip_if_active`` gives a *progressing* run no grace period at
    all, so one run stuck at ``running`` silently swallows every later firing
    of its schedule. Hence the belt-and-braces close-out at the end and the
    catch-all around the loop.
    """
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

    try:
        await _drive_workflow_steps(
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            model=model,
            project_name=project_name,
            project_path=project_path,
            launch_message=launch_message,
            steps=steps,
            permission_decision=permission_decision,
        )
    except Exception as exc:
        logger.exception(
            "workflow orchestration failed task_id=%s run_id=%s", task_id, run_id
        )
        _finish_workflow_execution(
            task_id=task_id,
            run_id=run_id,
            status="failed",
            error={
                "message": str(exc) or exc.__class__.__name__,
                "type": exc.__class__.__name__,
            },
        )
        return

    _close_out_unfinished_run(task_id=task_id, run_id=run_id)


def _close_out_unfinished_run(*, task_id: str, run_id: str) -> None:
    """Fail a run whose work stopped without anyone ending it.

    Reached only when a branch of the step loop returned without finishing or
    parking the run. That is a bug in the loop when it happens, but leaving the
    row at ``running`` turns that bug into a dead schedule, so close it here
    and say plainly that nothing is driving it any more.
    """
    store = get_agent_store()
    try:
        run = store.get_run(run_id)
    except Exception:
        logger.exception("could not re-read run %s to confirm it finished", run_id)
        return
    if not isinstance(run, dict) or run.get("status") not in _UNFINISHED_RUN_STATUSES:
        return
    logger.error(
        "workflow orchestration left run %s at %s with no work in progress; "
        "closing it as failed",
        run_id,
        run.get("status"),
    )
    _finish_workflow_execution(
        task_id=task_id,
        run_id=run_id,
        status="failed",
        error={
            "message": (
                "The workflow stopped without finishing; nothing is driving this run."
            )
        },
    )


#: A run parked on a person, in the statuses that park writes.
#:
#: The mirror of :data:`_UNFINISHED_RUN_STATUSES`: those are runs still being
#: driven, these are runs that stopped and are waiting. Only these can be
#: abandoned — a run that is working is never "unanswered".
_WAITING_RUN_STATUSES = {"blocked", "waiting_for_user", "waiting_user"}


#: Why a run ended when nobody ever answered the park it was sitting on.
#:
#: Deliberately *not* ``tool_call_denied``: there was nothing to deny. The run
#: stopped for a person — an ``ask_user`` failure policy, a manual handoff — and
#: no person came within the window the scheduler allows. Written on the step,
#: on the task's error and as its own run event, because "failed" on its own
#: tells the user nothing about why a run they never saw is red.
ABANDONED_WAIT_REASON = "abandoned_waiting_for_user"

_ABANDONED_WAIT_MESSAGE = (
    "Nobody answered this run, so it was ended and its schedule released."
)


def abandon_run_waiting_for_a_person(
    run_id: str,
    *,
    waited_seconds: float | None = None,
    wait_reason: str | None = None,
) -> dict[str, Any] | None:
    """End a run that parked for a person nobody ever sent.

    This is the *non-approval* half of abandonment. A run parked on an approval
    has something to answer and is settled through the deny path instead (see
    :func:`agent.approval_resume.abandon_waiting_run`); a run parked on
    ``ask_user`` or a manual handoff has no approval to deny, so there is
    nothing to hand the provider and the honest ending is to fail it *saying
    so*.

    Returns ``None`` when the run is no longer parked — the caller raced a
    resume, and a run that has started moving again must not be killed.
    """
    store = get_agent_store()
    try:
        run = store.get_run(run_id)
    except Exception:
        logger.exception("could not re-read run %s before abandoning it", run_id)
        return None
    if not isinstance(run, dict):
        return None
    status = run.get("status")
    if status not in _WAITING_RUN_STATUSES:
        # Somebody answered, or the run was already ended, between the decision
        # to abandon and this call.
        return None
    task_id = run.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return None

    checkpoint_context = store.get_task_checkpoint(task_id)
    step = (
        checkpoint_context.get("step") if isinstance(checkpoint_context, dict) else None
    )
    checkpoint = (
        checkpoint_context.get("checkpoint")
        if isinstance(checkpoint_context, dict)
        else None
    )
    if not wait_reason and isinstance(checkpoint, dict):
        raw_reason = checkpoint.get("reason")
        wait_reason = raw_reason if isinstance(raw_reason, str) else None

    error: dict[str, Any] = {
        "message": _ABANDONED_WAIT_MESSAGE,
        "reason": ABANDONED_WAIT_REASON,
    }
    if waited_seconds is not None:
        error["waited_seconds"] = int(waited_seconds)
    if wait_reason:
        # *What* went unanswered — `ask_user`, `manual_handoff`, … — so the
        # panel can say which question died rather than just that one did.
        error["waiting_for"] = wait_reason

    # A step can reach here having already been refused: a denied tool call
    # fails the step, `on_failure: ask_user` parks it again, and nobody answers
    # that park either. Writing this outcome over the step's output would drop
    # `denied_tool_calls`, and with it the only record of *what* was refused —
    # measured live on 2026-08-16, where a run ended saying nobody answered and
    # no longer said that Read had been denied on a named path. The refusal is
    # the more useful half of the story, so it is carried forward.
    previous_denials = _recorded_tool_denials(step) if isinstance(step, dict) else []
    if previous_denials:
        error[STEP_DENIED_TOOL_CALLS_KEY] = previous_denials

    store.append_event(
        run_id=run_id,
        event_type="task.run.abandoned",
        app_event={"task_id": task_id, "error": error},
    )

    if isinstance(step, dict) and step.get("run_id") in (None, run_id):
        _fail_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=str(step["id"]),
            run_id=run_id,
            error=error,
        )

    _finish_workflow_execution(
        task_id=task_id,
        run_id=run_id,
        status="failed",
        error=error,
    )
    return {"run_id": run_id, "task_id": task_id, "error": error}


async def _drive_workflow_steps(
    *,
    task_id: str,
    run_id: str,
    provider_id: str,
    model: str | None,
    project_name: str,
    project_path: str,
    launch_message: str,
    steps: list[dict[str, Any]],
    permission_decision: str | None = None,
) -> None:
    store = get_agent_store()
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
        # A user *message* completes a step that was waiting for a message. An
        # approval park is waiting for a decision on a specific tool call, and
        # marking it completed here would skip the tool call the run stopped
        # for and report the step as done without it ever having run.
        if (
            step.get("status") == "waiting_for_user"
            and _step_has_user_response(step)
            and _approval_checkpoint_for_step(step) is None
        ):
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
                steps = _steps_for_run(store, task_id, run_id)
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
                steps = _steps_for_run(store, task_id, run_id)
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
                steps = _steps_for_run(store, task_id, run_id)
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
        # The auto-advance loop dispatches separately from the single-step
        # path above. A type added to only one of them runs by hand and stalls
        # on a schedule, which is how `shell` first shipped.
        if workflow_type == "notify":
            completed = await _execute_notify_workflow_step(
                task_id=task_id, run_id=run_id, step=step
            )
            if not completed:
                next_index = _apply_workflow_failure_policy(
                    task_id=task_id,
                    run_id=run_id,
                    steps=steps,
                    failed_index=step_index,
                    error={
                        "message": f"Notify step '{step.get('title')}' did not send."
                    },
                )
                if next_index is None:
                    return
                steps = _steps_for_run(store, task_id, run_id)
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
            permission_decision=permission_decision,
        )
        # One decision answers one parked prompt. If the same turn parks again
        # on a different tool, that is a new question for the user, not a
        # second use of the answer they already gave.
        permission_decision = None
        # Parked on a person (None) is not a failed step: running the failure
        # policy here is what used to overwrite the approval checkpoint with an
        # `ask_user` one a second later.
        if completed is None:
            return
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
            steps = _steps_for_run(store, task_id, run_id)
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


#: What the model is told when the user refused the tool call it parked on.
#: It goes back through the SDK's `can_use_tool` result, so the turn keeps
#: running and the model gets to wrap up (or fail) knowing why.
_APPROVAL_DENIED_MESSAGE = (
    "The user denied this tool call. Do not retry it; finish with what you "
    "have and say plainly what you could not do."
)

#: Decision strings that mean "go ahead". `approvals/approval_models.py`
#: constrains the API to approve_once / approve_for_session / approve_rule /
#: deny / deny_with_instruction; the plain "approve"/"allow"/"deny" spellings
#: are accepted too so an internal caller does not have to know the API's
#: vocabulary to answer a parked turn.
_APPROVAL_ALLOW_DECISIONS = frozenset(
    {"allow", "approve", "approved", "approve_once", "approve_for_session", "approve_rule"}
)
_APPROVAL_DENY_DECISIONS = frozenset(
    {"deny", "denied", "deny_with_instruction", "reject", "rejected", "expired"}
)


#: What a run parked on a tool approval is called, everywhere. The workflow
#: path writes it as a checkpoint reason and the non-workflow path as an error
#: reason; both describe one situation and must say so with one word, because
#: the app groups and labels these by reason string.
APPROVAL_WAIT_REASON = "approval_required"


#: Where a step keeps the tool calls that were refused while it ran.
#:
#: It lives on the step's ``output`` rather than in a local variable because a
#: step can be denied one tool, park on a *second* one, and be resumed minutes
#: later in a fresh process. Only a persisted record survives that, and without
#: it the rule below would quietly stop holding for exactly the runs that took
#: the longest to finish.
STEP_DENIED_TOOL_CALLS_KEY = "denied_tool_calls"

#: Why a step with a denied tool call is failed rather than completed.
#:
#: The rule, deliberately fail-closed: **a workflow step in which any tool call
#: was refused ends `failed`.** The runtime cannot judge whether the model met
#: the step's `success_criteria` some other way — the only witness to that is
#: the model's own prose, and taking its word for it is the guess this codebase
#: refuses to make. The provider turn ending `subtype: success` says the turn
#: was clean, not that the work happened. Failing is also the recoverable
#: mistake: `on_failure` runs, which for the default `ask_user` puts the person
#: back in the loop, while a false "completed" tells them a refusal succeeded
#: and is never corrected. See RESULT_020.
_DENIED_STEP_MESSAGE = (
    "A tool call was denied while this step ran, so the step is not reported as "
    "completed."
)


def _recorded_tool_denials(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Denials this step already recorded, from an earlier park/resume cycle."""
    output = step.get("output")
    if not isinstance(output, dict):
        return []
    recorded = output.get(STEP_DENIED_TOOL_CALLS_KEY)
    if not isinstance(recorded, list):
        return []
    return [entry for entry in recorded if isinstance(entry, dict)]


def _merge_step_output(step: dict[str, Any], extra: dict[str, Any]) -> None:
    """Merge keys into a step's output, on the row *and* on the in-memory dict.

    Both halves matter. The row is what a later resume — possibly in a fresh
    process — will read. The in-memory dict is what ``_wait_for_user_step``
    builds its checkpoint output from: it takes the *passed-in* step, which the
    driver loop is still holding from before the failure, so anything written
    only to the row is dropped the moment an ``ask_user`` policy parks the step.
    """
    if not extra:
        return
    output = dict(step.get("output") or {})
    output.update(extra)
    step["output"] = output
    get_agent_store().update_task_step(step["id"], {"output": output})


def _record_tool_denials(step: dict[str, Any], denials: list[dict[str, Any]]) -> None:
    """Persist the tool calls this step has had refused so far."""
    if not denials:
        return
    _merge_step_output(step, {STEP_DENIED_TOOL_CALLS_KEY: denials})


def _denied_step_error(
    denials: list[dict[str, Any]], result_text: str | None
) -> dict[str, Any]:
    """The failure a denied step is recorded with.

    Carries the model's closing words as evidence: it was told to "say plainly
    what you could not do", and that sentence is the most useful thing on the
    step for whoever answers the `ask_user` this failure triggers.
    """
    error: dict[str, Any] = {
        "message": _DENIED_STEP_MESSAGE,
        "reason": "tool_call_denied",
        STEP_DENIED_TOOL_CALLS_KEY: denials,
    }
    if result_text:
        error["result"] = _truncate_workflow_evidence(result_text, 4000)
    return error


def _approval_checkpoint_for_step(step: dict[str, Any]) -> dict[str, Any] | None:
    """The step's own approval checkpoint, if it is parked on one right now."""
    if step.get("status") != "waiting_for_user":
        return None
    output = step.get("output")
    checkpoint = output.get("checkpoint") if isinstance(output, dict) else None
    if not isinstance(checkpoint, dict):
        return None
    if checkpoint.get("reason") != APPROVAL_WAIT_REASON:
        return None
    return checkpoint


def _approval_checkpoint_extra(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The approval identity to carry over when a checkpoint is re-written."""
    return {
        "approval_id": checkpoint.get("approval_id"),
        "tool_name": checkpoint.get("tool_name"),
        "tool_target": checkpoint.get("tool_target"),
    }


def _approval_tool_target(tool_input: Any) -> str | None:
    """The one concrete thing a parked tool call wants to touch.

    A checkpoint that says only "approval required" is not answerable from a
    phone. Tool inputs differ per tool, so read the handful of keys that name a
    target and take the first that is there; anything unrecognised gets no
    target rather than a guess.
    """
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "path", "command", "url", "pattern", "notebook_path", "query"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


_GENERIC_APPROVAL_PROMPT = "Approval is required before this workflow step can continue."


def _approval_wait_prompt(tool_name: Any, tool_target: Any) -> str:
    """The sentence a phone shows for a run parked on a tool approval.

    "A task is waiting for you" is not answerable from a lock screen — the
    question is always *which tool wants to touch what*, and both are already
    on the checkpoint. Falls back to the generic line only when the tool call
    named neither (an unrecognised tool input, see `_approval_tool_target`),
    because a half-built sentence reads worse than a plain one.
    """
    name = tool_name.strip() if isinstance(tool_name, str) else ""
    target = tool_target.strip() if isinstance(tool_target, str) else ""
    if name and target:
        return f"{name} needs your approval: {target}"
    if name:
        return f"{name} needs your approval before this step can continue."
    return _GENERIC_APPROVAL_PROMPT


def _resolve_permission_decision(
    checkpoint: dict[str, Any],
    permission_decision: str | None,
) -> str | None:
    """``"allow"``, ``"deny"``, or ``None`` when nobody has decided yet.

    An explicit decision (the approval endpoint handing over what the user just
    pressed) wins. Otherwise the approval row itself is the record of truth —
    a manual resume from the app carries no decision, and the run must not
    invent one.
    """
    explicit = str(permission_decision or "").strip().lower()
    if explicit in _APPROVAL_ALLOW_DECISIONS:
        return "allow"
    if explicit in _APPROVAL_DENY_DECISIONS:
        return "deny"

    approval_id = checkpoint.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id:
        return None
    try:
        approval = get_approval_store().get_request(approval_id)
    except Exception:
        logger.exception("could not read approval %s while resuming", approval_id)
        return None
    if not isinstance(approval, dict):
        return None
    status = str(approval.get("status") or "").strip().lower()
    if status == "approved":
        return "allow"
    if status in {"denied", "expired"}:
        return "deny"
    if status == "pending" and is_request_expired(approval):
        # Past its deadline but the sweep has not reached it yet. Treat it as
        # the refusal it is — and write that down first, so the row agrees with
        # what the run is about to do instead of the run quietly denying an
        # approval the queue still shows as answerable.
        try:
            expire_approval(approval_id, trigger="resume")
        except Exception:
            logger.exception("could not expire approval %s while resuming", approval_id)
        return "deny"
    return None


def _parked_permission_session(session_scope: str) -> Any | None:
    """The cached provider session parked on an unanswered permission prompt.

    ``None`` means there is nothing to continue — either the server restarted
    and the session is gone, or the provider does not park turns at all (only
    Claude does) — and the caller falls back to re-executing the step.
    """
    try:
        session = get_session_manager().get_session_if_exists(session_scope)
    except Exception:
        logger.exception("could not look up parked session %s", session_scope)
        return None
    if session is None:
        return None
    return session if getattr(session, "has_pending_permission", False) else None


#: The only provider whose sessions can be handed MCP server configs. The
#: Claude Agent SDK is where the option lives (`ClaudeAgentOptions.mcp_servers`,
#: claude-agent-sdk 0.2.128); codex, gemini and antigravity sessions have
#: nowhere to put one, so a declared server on those providers is reported as
#: not injected rather than assumed to work.
MCP_INJECTION_PROVIDER_ID = "anthropic"

#: Event type carrying what happened to an agent's declared MCP tools on one
#: turn. Emitted whenever an agent declares any, so the run log answers "was
#: the tool this agent promises actually there?" without a second lookup.
MCP_TOOLS_EVENT_TYPE = "task.step.mcp_tools"


def _declared_mcp_tools(agent: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The ``tools_json`` entries of an agent, keyed by ``mcp_id``.

    First entry wins for a repeated id, matching :func:`_declared_mcp_ids`,
    which reports each declaration once in first-seen order.
    """
    tools = (agent or {}).get("tools_json")
    if not isinstance(tools, list):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        mcp_id = tool.get("mcp_id")
        if isinstance(mcp_id, str) and mcp_id.strip():
            entries.setdefault(mcp_id.strip(), tool)
    return entries


def _declared_mcp_ids(agent: dict[str, Any] | None) -> list[str]:
    """The ``mcp_id`` values in an agent's ``tools_json``, in order."""
    tools = (agent or {}).get("tools_json")
    if not isinstance(tools, list):
        return []
    declared: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        mcp_id = tool.get("mcp_id")
        if isinstance(mcp_id, str) and mcp_id.strip():
            declared.append(mcp_id.strip())
    return declared


#: Which workflow step types actually run on which declared runtime. Only the
#: step *type* counts: `configurator._add_playwright_hints` stamps
#: `tool_hint: "playwright"` onto any step whose prose merely mentions a URL or
#: the word "click" (configurator.py:2048), so treating a `tool_hint` on an
#: ordinary `llm` step as proof of need would just re-run the same keyword
#: guess the need check exists to stop trusting.
_STEP_TYPES_REQUIRING_MCP_ID: dict[str, frozenset[str]] = {
    # A `browser_action` step drives the server's built-in Playwright runtime,
    # and its flow is a flow that automates a browser: an llm step in it
    # reaching for the `playwright` MCP server is the agent doing its job.
    "playwright": frozenset({"browser_action"}),
    BROWSER_RUNTIME_CAPABILITY_NAME: frozenset({"browser_action"}),
    # Same four types `_is_app_action_workflow_type` dispatches on; the
    # agreement is asserted in tests/test_mcp_injection_need.py.
    "app_action": frozenset(
        {"app_action", "android_action", "mobile_action", "device_action"}
    ),
}

#: `tool_hint` values on an ``mcp_tool`` step that name a runtime rather than
#: repeating the server id. An ``mcp_tool`` step is the author saying "call
#: this server here", which is the strongest need signal there is, so the
#: aliases the builder writes for those steps have to resolve too.
_MCP_TOOL_HINT_ALIASES: dict[str, frozenset[str]] = {
    "playwright": frozenset({"playwright", "browser", "browser_action"}),
    "app_action": frozenset(
        {
            "app_action",
            "android_adb",
            "android",
            "android-device",
            "android_device",
            "mobile_action",
            "device_action",
        }
    ),
}


def _agent_flow_steps(agent: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The agent's stored workflow steps, raw.

    Raw rather than normalized on purpose: a flow that `normalize_workflow`
    would reject still says which step types it contains, and refusing to read
    it would silently turn "this agent has a browser step" into "this agent
    needs nothing".
    """
    flow = (agent or {}).get("flow_json")
    if not isinstance(flow, list):
        return []
    return [step for step in flow if isinstance(step, dict)]


def _flow_need_for_mcp_id(flow: list[dict[str, Any]], mcp_id: str) -> str | None:
    """The step that needs ``mcp_id``, described, or ``None`` if none does."""
    aliases = _MCP_TOOL_HINT_ALIASES.get(mcp_id, frozenset()) | {mcp_id.casefold()}
    step_types = _STEP_TYPES_REQUIRING_MCP_ID.get(mcp_id, frozenset())
    for index, step in enumerate(flow):
        raw_type = step.get("type") or step.get("step_type") or "llm"
        step_type = str(raw_type).strip().casefold()
        name = str(step.get("name") or step.get("id") or f"step {index + 1}").strip()
        if step_type == "mcp_tool":
            hint = str(step.get("tool_hint") or "").strip().casefold()
            if hint and hint in aliases:
                return f'step "{name}" calls it directly'
        if step_type in step_types:
            return f'step "{name}" is a {step_type} step'
    return None


def _mcp_id_is_needed(
    mcp_id: str,
    *,
    declared_entry: dict[str, Any] | None,
    flow: list[dict[str, Any]],
) -> str | None:
    """Why this run needs ``mcp_id``, or ``None`` when nothing says it does.

    Two things count as need, and nothing else does:

    * the agent's own declaration — an entry the model wrote into the draft
      because the agent is supposed to use that server; and
    * a workflow step that runs on it.

    The gap between them is the whole point. `configurator._ensure_playwright_
    tool` adds a `playwright` entry to any draft whose *wording* looked web-ish
    ("click", "form", "url", "웹"), so on a machine with a `playwright` MCP
    server configured, an agent that never opens a browser was handed one —
    and, since C6-2, spawned `npx @playwright/mcp@latest` on every llm turn,
    every two minutes for a scheduled agent. A builder-added entry with no step
    to justify it is therefore not a need; a model-declared entry always is.
    """
    if not is_builder_added_tool(declared_entry):
        # Includes every agent stored before this check existed whose entry
        # does not match the builder's templates: unrecognised means
        # "the model declared it", so those runs keep the server they have
        # been getting instead of quietly losing it.
        return "the agent declares it"
    return _flow_need_for_mcp_id(flow, mcp_id)


async def _apply_declared_mcp_servers(
    session: Any,
    *,
    agent_id: str | None,
    run_id: str,
    task_id: str,
    provider_id: str,
    step_id: str | None = None,
) -> None:
    """Give the session the MCP servers this agent declared, and say what it could not.

    Until this existed an agent's ``tools_json`` had no run-time consumer at
    all: a declared server was a row in the database, the turn ran on whatever
    the user's CLI was configured with, and nothing anywhere reported the
    difference. The two halves here are equally load-bearing — servers that do
    exist are passed through to :meth:`ClaudeSession.set_mcp_servers`, and
    every declaration that could *not* be passed through is written to the run
    as :data:`MCP_TOOLS_EVENT_TYPE`. Dropping the second half would restore the
    exact silence this replaces.

    Never raises for the missing case: a declared-but-absent server is
    reported, not fatal, because the turn may well not need it, and failing
    the step would take a working run down over a tool it never calls.

    A configured server is only passed through when this agent actually needs
    it — see :func:`_mcp_id_is_needed`. One that exists but nothing in the
    agent asks for is reported as ``not_injected`` with
    ``reason: "not_required_by_agent"``, never as ``missing``: it is here, it
    was simply not started, and the two are different facts.
    """
    store = get_agent_store()
    agent = store.get_agent(agent_id) if agent_id else None
    declared = _declared_mcp_ids(agent)
    if not declared:
        return

    verdicts = verify_declared_mcp_ids(declared)
    configs = detected_mcp_server_configs()
    declared_entries = _declared_mcp_tools(agent)
    flow = _agent_flow_steps(agent)

    builtin: list[str] = []
    missing: list[dict[str, str]] = []
    unlaunchable: list[dict[str, str]] = []
    injectable: dict[str, Any] = {}
    for verdict in verdicts:
        if verdict.source == "builtin_runtime":
            # Not gated by need: a built-in runtime starts no process for this
            # agent and injects nothing, so there is nothing to withhold, and
            # reporting it as "not required" instead of "runs on our own
            # runtime" would replace a true sentence with a misleading one.
            builtin.append(verdict.mcp_id)
            continue
        if not verdict.verified:
            missing.append({"mcp_id": verdict.mcp_id, "detail": verdict.detail})
            continue
        config = configs.get(verdict.mcp_id)
        if config is None:
            unlaunchable.append(
                {
                    "mcp_id": verdict.mcp_id,
                    "reason": "no_launch_config",
                    "detail": (
                        f'MCP server "{verdict.mcp_id}" is configured but its entry does '
                        "not describe how to launch it (no command for a stdio server, "
                        "no url for an http/sse server), so it was not passed to the run."
                    ),
                }
            )
            continue
        # Asked before the session is consulted, because need is a property of
        # the agent rather than of the provider: a server this agent has no
        # use for should read as "not required" on every provider, not as
        # "your session type could not take it".
        need = _mcp_id_is_needed(
            verdict.mcp_id,
            declared_entry=declared_entries.get(verdict.mcp_id),
            flow=flow,
        )
        if need is None:
            unlaunchable.append(
                {
                    "mcp_id": verdict.mcp_id,
                    "reason": "not_required_by_agent",
                    "detail": (
                        f'MCP server "{verdict.mcp_id}" is configured on this machine, but '
                        "the Agent Builder added this tool from the wording of the request "
                        "rather than the agent asking for it, and no workflow step uses it, "
                        "so this run does not start it. Give the agent a step that uses it "
                        f'(a browser_action step, or an mcp_tool step naming "{verdict.mcp_id}") '
                        "if it really needs it."
                    ),
                }
            )
            continue
        # Only the name is kept for the event; the config itself carries env
        # and headers, which can hold credentials.
        injectable[verdict.mcp_id] = config

    if provider_id != MCP_INJECTION_PROVIDER_ID:
        blocked_by = (
            f"the '{provider_id}' session type has no MCP server option "
            f"(only '{MCP_INJECTION_PROVIDER_ID}' does)"
        )
    elif not hasattr(session, "set_mcp_servers"):
        blocked_by = f"this {type(session).__name__} session cannot be given MCP servers"
    else:
        blocked_by = ""

    supported = not blocked_by
    injected: list[str] = []
    if supported:
        if injectable:
            await session.set_mcp_servers(injectable)
            injected = sorted(injectable)
    elif injectable:
        # The servers are here, the session cannot take them. Say so rather
        # than letting the turn look as though it had them.
        unlaunchable.extend(
            {
                "mcp_id": name,
                "reason": "session_cannot_take_mcp_servers",
                "detail": (
                    f'MCP server "{name}" is configured on this machine, but '
                    f"{blocked_by}, so this run does not have it."
                ),
            }
            for name in sorted(injectable)
        )

    notes: list[str] = []
    if injected:
        notes.append(f"Passed to this run: {', '.join(injected)}.")
    if builtin:
        notes.append(
            f"Runs on this server's own runtime, not an MCP server: {', '.join(builtin)}."
        )
    for entry in (*missing, *unlaunchable):
        notes.append(entry["detail"])

    store.append_event(
        run_id=run_id,
        event_type=MCP_TOOLS_EVENT_TYPE,
        provider_id=provider_id,
        app_event={
            "task_id": task_id,
            "step_id": step_id,
            "agent_id": agent_id,
            "declared": declared,
            "injected": injected,
            "builtin_runtime": builtin,
            "missing": missing,
            "not_injected": unlaunchable,
            "provider_supports_mcp_injection": supported,
            "message": " ".join(notes),
        },
    )


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
    permission_decision: str | None = None,
) -> bool | None:
    """Run one llm workflow step.

    Tri-state, exactly like the browser and app-action executors: ``True`` the
    step finished, ``False`` it failed and the caller should apply the step's
    failure policy, and ``None`` the run is *parked on a person* and nothing
    failed. Returning ``False`` for an approval pause — which is what this did
    until 2026-08-16 — made the driver run ``on_failure``, whose ``ask_user``
    branch immediately overwrote the approval checkpoint with its own; the
    phone then showed two ``waiting_for_user`` events a second apart and the
    approval the user was asked for was no longer the one the run was waiting
    on.
    """
    store = get_agent_store()
    step_id = str(step["id"])
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    agent_id = _step_agent_id(store.get_task(task_id) or {})
    session_scope = f"task:{task_id}"

    # --- Resuming a step that is parked on an approval --------------------
    # The provider session that asked for permission is still alive with the
    # SDK's `can_use_tool` callback parked on a future, so the honest resume is
    # to answer *that* callback and let the same turn carry on. Re-sending the
    # step message instead would hit "Another Claude turn is already in
    # progress" (llm/claude_session.py send_message), which is what every
    # resume did before this branch existed.
    approval_checkpoint = _approval_checkpoint_for_step(step)
    parked_session: Any | None = None
    resume_decision: str | None = None
    # Denials this step already collected before it parked. See
    # `STEP_DENIED_TOOL_CALLS_KEY`: a refusal earlier in the step still counts
    # against the step when a later tool is approved and the turn ends clean.
    denials: list[dict[str, Any]] = _recorded_tool_denials(step)
    if approval_checkpoint is not None:
        resume_decision = _resolve_permission_decision(
            approval_checkpoint, permission_decision
        )
        if resume_decision is None:
            # Nobody has decided yet. Re-park rather than run anything: the
            # caller already flipped the run to `running`, and leaving it there
            # would have `_close_out_unfinished_run` fail a run that is simply
            # waiting for its owner. Same run/step/reason means
            # `_wait_for_user_step` recognises the repeat and does not notify
            # again.
            _wait_for_user_step(
                task_id=task_id,
                run_id=run_id,
                step=step,
                reason=APPROVAL_WAIT_REASON,
                prompt=(
                    approval_checkpoint.get("prompt")
                    if isinstance(approval_checkpoint.get("prompt"), str)
                    else None
                ),
                checkpoint_extra=_approval_checkpoint_extra(approval_checkpoint),
            )
            store.append_event(
                run_id=run_id,
                event_type="task.step.approval_still_pending",
                provider_id=provider_id,
                app_event={
                    "task_id": task_id,
                    "step_id": step_id,
                    "approval_id": approval_checkpoint.get("approval_id"),
                },
            )
            return None
        if resume_decision == "deny":
            # The decision itself is the denial record — deterministic, owned by
            # the runtime, and written before the turn that will be told about
            # it. Nothing downstream has to read the model's prose to know a
            # tool call was refused in this step.
            denials.append(
                {
                    "source": "approval_decision",
                    "approval_id": approval_checkpoint.get("approval_id"),
                    "tool_name": approval_checkpoint.get("tool_name"),
                    "tool_target": approval_checkpoint.get("tool_target"),
                    "decision": str(permission_decision or "").strip().lower() or "deny",
                }
            )
            _record_tool_denials(step, denials)
        parked_session = _parked_permission_session(session_scope)
        if parked_session is None and resume_decision == "deny":
            # Re-running the step would ask the model to do the very thing the
            # user just refused. Fail it instead and let the step's failure
            # policy decide, rather than quietly doing the denied work.
            _fail_step(
                task=store.get_task(task_id) or {"id": task_id},
                step_id=step_id,
                run_id=run_id,
                error={
                    "message": (
                        "Approval was denied and the provider session that asked "
                        "for it is gone, so the step was not re-run."
                    ),
                    "reason": "tool_call_denied",
                    "approval_id": approval_checkpoint.get("approval_id"),
                    STEP_DENIED_TOOL_CALLS_KEY: denials,
                },
            )
            return False
        store.append_event(
            run_id=run_id,
            event_type="task.step.approval_resumed",
            provider_id=provider_id,
            app_event={
                "task_id": task_id,
                "step_id": step_id,
                "approval_id": approval_checkpoint.get("approval_id"),
                "decision": resume_decision,
                # Which of the two resume paths ran, so a run log says whether
                # the original turn continued or the step was started over.
                "mode": "same_turn" if parked_session is not None else "re_execute",
            },
        )

    matched_memories = _memories_for_step(store, agent_id, step_input)
    store.update_task_step(step_id, {"status": "running"})
    store.append_event(
        run_id=run_id,
        event_type="task.step.started",
        provider_id=provider_id,
        app_event={"task_id": task_id, "step_id": step_id, "mode": "workflow"},
    )
    sink = AgentTaskRunSink(run_id=run_id)
    try:
        if parked_session is not None:
            if resume_decision == "deny":
                completed = await stream_claude_turn(
                    sink,
                    parked_session,
                    project_name=project_name,
                    deny_from_permission_message=_APPROVAL_DENIED_MESSAGE,
                )
            else:
                completed = await stream_claude_turn(
                    sink,
                    parked_session,
                    project_name=project_name,
                    retry_from_permission=True,
                )
        else:
            # Read the agent-definition file, if this agent is backed by one,
            # before the session exists: a missing or broken source must fail the
            # step naming the file, not start a turn that runs something else. Both
            # this and the wrong-provider refusal below raise, and the handler
            # underneath records the message on the step.
            cli_agent = resolve_cli_agent_definition(agent_id)
            session = await create_chat_session(
                project_name=session_scope,
                project_path=project_path,
                selection=ChatProviderSelection(
                    provider_id=provider_id,
                    provider_name=_provider_name(provider_id),
                    model=model,
                ),
                cli_agent=cli_agent,
            )
            # Before the turn, so the servers are in the options this session
            # connects with. A change here closes an already-connected session,
            # which is why it cannot wait until after the first message.
            await _apply_declared_mcp_servers(
                session,
                agent_id=agent_id,
                run_id=run_id,
                task_id=task_id,
                provider_id=provider_id,
                step_id=step_id,
            )
            completed = await stream_claude_turn(
                sink,
                session,
                project_name=project_name,
                user_message=_workflow_step_message(
                    step,
                    launch_message,
                    previous_steps=previous_steps or [],
                    matched_memories=matched_memories,
                    cli_agent_source_path=cli_agent.source_path if cli_agent else None,
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

    # Refusals the turn itself carried (a standing rule denying a tool never
    # stops the turn, so this is the only place they are seen).
    denials.extend(sink.denied_tool_calls)

    if sink.permission_required:
        # Parked again, on a different tool. The step is not over, so the
        # denials it has collected so far have to outlive the park.
        _record_tool_denials(step, denials)
        denial = sink.permission_denial or {}
        tool_name = denial.get("tool_name")
        tool_target = _approval_tool_target(denial.get("input"))
        _wait_for_user_step(
            task_id=task_id,
            run_id=run_id,
            step=step,
            reason=APPROVAL_WAIT_REASON,
            # Names the tool and its target, so the push says "Read needs your
            # approval: /etc/hosts" instead of a line that could be about any
            # step of any run.
            prompt=_approval_wait_prompt(tool_name, tool_target),
            checkpoint_extra={
                "approval_id": denial.get("approval_id"),
                "tool_name": tool_name,
                "tool_target": tool_target,
            },
        )
        return None
    if sink.error_message or not completed:
        error: dict[str, Any] = {
            "message": sink.error_message or "Provider turn did not complete."
        }
        if denials:
            error[STEP_DENIED_TOOL_CALLS_KEY] = denials
        _fail_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            error=error,
        )
        return False

    if denials:
        # The turn ended cleanly, and that is not the same thing as the step
        # having done its work: a `subtype: success` result only says the
        # provider finished. Something in this step was refused, so the step
        # failed and `on_failure` decides what happens next — which is how the
        # model's own "I could not do this" reaches the person.
        denied_error = _denied_step_error(denials, sink.result_text)
        # Onto the step before it fails, so an `ask_user` park — which rebuilds
        # the output from the step the driver loop is holding — keeps the reason
        # and the model's account instead of showing a prompt with no cause.
        _merge_step_output(step, denied_error)
        _fail_step(
            task=store.get_task(task_id) or {"id": task_id},
            step_id=step_id,
            run_id=run_id,
            error=denied_error,
        )
        return False

    output: dict[str, Any] = {"message": "Workflow step completed."}
    if sink.result_text:
        # Truncated: a step output row is read in a list, and the full turn is
        # still in the event log if anyone needs all of it.
        output["result"] = _truncate_workflow_evidence(sink.result_text, 4000)
    _remember_step_result(
        store,
        agent_id=agent_id,
        run_id=run_id,
        step_input=step_input,
        result_text=sink.result_text,
    )
    _complete_step(
        task=store.get_task(task_id) or {"id": task_id},
        step_id=step_id,
        run_id=run_id,
        output=output,
    )
    return True


async def _execute_notify_workflow_step(
    *,
    task_id: str,
    run_id: str,
    step: dict[str, Any],
) -> bool:
    """Leave a message the phone can read later.

    A scheduled run ends while nobody is watching, so the useful part of "tell
    me if the disk is nearly full" is what survives the run. Earlier steps put
    their findings on the step record; this one turns the sentence the
    workflow was given into something the inbox will show.
    """
    from agent.notification_store import get_notification_store, normalize_level

    store = get_agent_store()
    step_id = str(step["id"])
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}

    store.update_task_step(step_id, {"status": "running"})

    payload = step_input.get("notify") if isinstance(step_input.get("notify"), dict) else {}
    title = str(payload.get("title") or step.get("title") or "").strip()
    if not title:
        error = {
            "message": "notify step has no title to send",
            "type": "NotificationTitleMissing",
        }
        store.update_task_step(step_id, {"status": "failed", "output": {"error": error}})
        return False

    task = store.get_task(task_id) or {}
    notify_body = payload.get("body") or step_input.get("description") or None
    notify_level = normalize_level(payload.get("level"))
    try:
        notification = get_notification_store().create(
            title=title,
            body=notify_body,
            level=notify_level,
            run_id=run_id,
            task_id=task_id,
            agent_id=task.get("assigned_agent_id"),
        )
    except (ValueError, RuntimeError, OSError) as exc:
        store.update_task_step(
            step_id,
            {
                "status": "failed",
                "output": {"error": {"message": str(exc), "type": type(exc).__name__}},
            },
        )
        return False

    _push_notification_best_effort(
        notification=notification,
        title=title,
        body=notify_body,
        level=notify_level,
    )

    store.update_task_step(
        step_id,
        {"status": "completed", "output": {"notification": notification}},
    )
    return True


def _push_notification_best_effort(
    *,
    notification: dict[str, Any],
    title: str,
    body: str | None,
    level: str,
) -> None:
    """Best-effort FCM push for a notification that is already durably stored.

    The step already succeeded by the time this runs — the notification is
    in the store and the inbox will show it on the next pull regardless. A
    push failure (no key configured, FCM rejects the send, network error) is
    a lost ring, not a lost message, so it is logged and swallowed here
    rather than allowed to fail the step.
    """
    try:
        from pairing.pairing import get_pairing_service

        tokens = get_pairing_service().all_push_tokens()
        if not tokens:
            return

        from agent.push_notifier import send_to_tokens

        result = send_to_tokens(
            tokens,
            title=title,
            body=body,
            notification_id=notification.get("id"),
            level=level,
        )
        for dead_token in result.get("dropped", []):
            get_pairing_service().remove_push_token(dead_token)
    except Exception:
        logger.exception("Push notification attempt failed; notification is still stored")


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


def _bindings_from_earlier_steps(
    store: Any, *, run_id: str, task_id: str, step_id: str
) -> dict[str, str]:
    """Values earlier browser steps of this run named, for this step to use.

    A workflow cannot contain the ids a site only reveals at runtime, so a flow
    finds one in one step — "read the cafe id out of the page" — and uses it in
    the next. Each step runs its own adapter call, so without this the value
    died at the step boundary and the next step parked on the `{{name}}` the
    run had already answered.

    Same run only. Yesterday's id is not this run's evidence, and silently
    reusing it would send the step somewhere nobody looked at.
    """
    bindings: dict[str, str] = {}
    try:
        steps = store.list_task_steps(task_id) or []
    except Exception:  # noqa: BLE001 - a missing binding parks, it never crashes
        logger.debug("could not read earlier steps for bindings", exc_info=True)
        return bindings

    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("run_id") != run_id or step.get("id") == step_id:
            continue
        output = step.get("output")
        if not isinstance(output, dict):
            continue
        browser = output.get("browser_action")
        if not isinstance(browser, dict):
            continue
        for found in browser.get("extracted") or []:
            if not isinstance(found, dict):
                continue
            name = found.get("name")
            value = found.get("value")
            # Later steps win: a flow may re-read a value that changed, and the
            # freshest reading is the one the run just made.
            if isinstance(name, str) and name and value is not None:
                bindings[name] = str(value)
    return bindings


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
        task_id=task_id,
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
            "bindings": _bindings_from_earlier_steps(
                store, run_id=run_id, task_id=task_id, step_id=step_id
            ),
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
    # Same resolver the context below uses, so the session's recorded
    # provenance cannot disagree with the state actually handed to the adapter.
    previous = _previous_browser_session_for_execution(
        run_id=run_id,
        task_id=task_id,
        current_session_id=None,
    )
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
    task_id: str | None,
    current_session_id: str | None,
) -> dict[str, Any] | None:
    """Which earlier browser session does this step continue from?

    The newest session of the same *task* that left a storage state behind,
    whichever run it belonged to. That single rule covers both cases:

    *   an earlier **step in this run** — newer than anything before it, so it
        wins, which is the behaviour this already had;
    *   the previous **run of the same task** — a scheduled agent gets a new
        ``run_id`` every night, and this used to look no further than the
        current run (``latest_resumable_for_run``), so every nightly run
        started signed out however many times a person had logged in. The task
        id is what survives between runs.

    Only the current session is excluded, never the whole current run:
    excluding the run would also skip that earlier step. That exclusion is
    what stops a session shadowing a real previous state with the
    ``storage_state.json`` it has not written yet.

    The lookup lives here rather than in `_browser_session_context` because
    deciding *which* session to continue from is this function's whole job;
    `_browser_session_context` only formats a session pair into the adapter's
    context. Putting a store query there would give the formatter a second
    responsibility and let the session id and the storage-state path in the
    same dict come from two different lookups.
    """
    store = get_browser_session_store()
    if not task_id:
        # No task to scope by (direct/ad-hoc execution): the run is all there
        # is to go on, which is the pre-existing behaviour.
        previous = store.latest_resumable_for_run(run_id)
        if previous is not None and previous.get("id") != current_session_id:
            return previous
        return None
    return store.latest_with_storage_state_for_task(
        task_id, exclude_session_id=current_session_id
    )


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
        step
        for step in _steps_for_run(store, task_id, run_id)
        if step.get("status") == "failed"
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
            # A retry is a *fresh attempt*, so it starts with no denials against
            # it. The record exists to survive a park inside one attempt (see
            # `STEP_DENIED_TOOL_CALLS_KEY`); carrying it into the next attempt
            # would fail an attempt in which nothing was refused, which is a lie
            # in the other direction. The error it came from is preserved above
            # in `last_retry_error`.
            output.pop(STEP_DENIED_TOOL_CALLS_KEY, None)
            failed_step["output"] = output
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
    if (
        not prompt
        and reason == APPROVAL_WAIT_REASON
        and isinstance(checkpoint_extra, dict)
        and checkpoint_extra.get("tool_name")
    ):
        # Any caller that parks on an approval with the tool identity in hand
        # gets the tool-specific sentence, whether or not it remembered to pass
        # one. `_checkpoint_prompt`'s fallback names the step, which is what
        # every approval notification used to say.
        prompt = _approval_wait_prompt(
            checkpoint_extra.get("tool_name"), checkpoint_extra.get("tool_target")
        )
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
    previous_checkpoint = metadata.get("active_checkpoint")
    new_checkpoint_marker = {
        "run_id": run_id,
        "step_id": step["id"],
        "workflow_step_id": checkpoint.get("workflow_step_id"),
        "reason": reason,
    }
    metadata["active_checkpoint"] = new_checkpoint_marker
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

    # Same run, same step, same reason as the last park means this is a
    # resume-then-immediately-re-park loop (e.g. the user resumed a browser
    # handoff but the page is still showing the login form) rather than a
    # new event the user has not already been told about. Notifying again on
    # every such loop would turn one stuck run into a stream of pings.
    is_repeat_park = (
        isinstance(previous_checkpoint, dict)
        and previous_checkpoint.get("run_id") == run_id
        and previous_checkpoint.get("step_id") == step["id"]
        and previous_checkpoint.get("reason") == reason
    )
    if not is_repeat_park:
        _notify_waiting_for_user_best_effort(
            store=store,
            task=task,
            run_id=run_id,
            step=step,
            reason=reason,
            prompt=checkpoint.get("prompt") if isinstance(checkpoint.get("prompt"), str) else None,
        )


_WAIT_REASON_LABELS: dict[str, str] = {
    "login_required": "needs a login",
    "captcha_or_bot_challenge": "hit a CAPTCHA or bot check",
    "manual_handoff": "needs a manual handoff",
    "mcp_tool": "needs a tool/MCP confirmation",
    "approval_gate": "needs your approval",
    "approval_required": "needs your approval",
    "ask_user": "needs your input",
    "browser_action": "needs help with a browser step",
    "app_action": "needs help with a device action",
}


def _wait_reason_label(reason: str) -> str:
    """Turn a wait-reason code into a short phrase for a notification title.

    Known reasons get a human phrase; anything else (there are dozens of
    finer-grained `wait_reason` values from the browser/app adapters, e.g.
    ``browser_action_click_requires_review``) still gets read as words rather
    than a code, because a notification with no clue why it fired is not
    actionable at 3am.
    """
    label = _WAIT_REASON_LABELS.get(reason)
    if label:
        return label
    humanized = str(reason or "").replace("_", " ").strip()
    return humanized or "needs your attention"


# --- Notification throttling ------------------------------------------------
#
# Two scheduled agents on the machine this was built for failed the same way
# every six hours for days — a UI automation script that stopped finding a
# button, nothing this server could fix — and the phone got the same push
# four times a day, every day. That is worse than useless: it trains the
# reflex of swiping a notification away without reading it, and the swipe
# that eats today's fourth "X failed again" is the same reflex that eats the
# one push that actually mattered, like a run parked waiting for a human or
# an agent's source file gone missing.
#
# `_failure_notification_gate` and `_wait_notification_gate` below share one
# rule: a *first* occurrence always notifies (nobody should wait a day to
# learn their agent just broke), and a repeat of the *same* agent's ongoing
# trouble is throttled to once per `_NOTIFICATION_THROTTLE_WINDOW`. Both are
# best-effort in the same sense the notifications themselves are: a gate that
# cannot be evaluated fails **open** (notifies) rather than closed
# (suppresses), because a duplicate push during a rare DB hiccup costs far
# less than a genuine first failure going out silently — see each function's
# docstring for the specific reasoning.

#: How long a repeat of the same agent's ongoing trouble stays quiet after
#: the last notification about it. 24h matches the user's ask ("at most once
#: per day"); it is not meant to be tuned per-deployment, so unlike
#: `agent.cli_agent_sweep`'s interval this has no environment override.
_NOTIFICATION_THROTTLE_WINDOW = timedelta(hours=24)


def _parse_notification_instant(value: Any) -> datetime | None:
    """Parse a notification's ``created_at`` back into an aware UTC instant.

    ``NotificationStore`` always sends timestamps through
    ``core.timestamps.to_utc_iso`` before handing them out, so this only ever
    needs to parse an ISO-8601 string that may or may not already carry an
    offset. Anything else — ``None``, a blank string, a value that fails to
    parse — comes back as ``None`` rather than raising, so a corrupt or
    unexpected stamp reads as "can't tell", which callers here treat as a
    reason to notify rather than a reason to crash.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _most_recent_other_run(
    runs: list[dict[str, Any]], *, exclude_run_id: str
) -> dict[str, Any] | None:
    """The most recently touched run in ``runs`` other than ``exclude_run_id``.

    Used to look one run back in an agent's history to ask "did the run
    before this one succeed, or was it also in trouble". ``runs`` is expected
    to already be ``AgentStore.list_runs``'s own order (most recent first),
    so this just takes the first entry that isn't the run in hand — it does
    *not* re-derive recency from ``created_at``/``updated_at`` strings
    itself. SQLite's ``CURRENT_TIMESTAMP`` has only one-second resolution, so
    two runs of the same agent created or finished inside the same second
    (exactly what happens when a test — or a fast-failing script — fires a
    schedule back-to-back) tie on both timestamp columns; ``list_runs``
    breaks that tie with ``rowid DESC``, true insertion order, which a
    second read of the timestamp strings here cannot reconstruct.
    """
    for run in runs:
        if run.get("id") != exclude_run_id:
            return run
    return None


def _failure_notification_gate(
    *, store: Any, agent_id: str | None, run_id: str
) -> tuple[bool, str]:
    """Should *this* failed run push, or is it a quiet repeat of a known streak.

    The rule the user asked for: while an agent is failing consecutively,
    notify at most once per day. A *first* failure always notifies. A later
    failure of the *same* agent stays quiet if the agent's last failure
    notification is less than 24h old — but only while the streak is
    unbroken. A run that *succeeds* ends the streak: if an agent broke, got
    fixed, and broke again, that is two separate events to whoever is holding
    the phone, and suppressing the second because the first was recent would
    hide a regression behind its own quiet window. Concretely: this looks at
    the run immediately before the one that just failed, and treats a streak
    as unbroken only when that run also failed.

    Keyed on the agent alone, not on the agent plus some hash of the error
    text. Two reasons. First, "consecutive" was the user's own word, and it
    describes the run's *outcome* (did it fail again), not why — a script
    that fails for a new reason today is still the same broken agent
    training the same swipe-it-away reflex the user is trying to stop.
    Second, a signature built from a free-text error message would have to
    strip every incidental detail (a timestamp, a duration, a path) to mean
    anything at all, and a signature that changes on every run of the same
    broken script throttles nothing — which is precisely the failure mode
    the user warned against.

    A throttle-state read failure (run history or notification history)
    fails **open**: this returns "notify" rather than "suppress". The
    alternative — going quiet because a read broke — can swallow a genuine
    first failure without a trace, which is the exact harm this function
    exists to prevent; an extra push during a rare DB hiccup is a much
    cheaper mistake.
    """
    if not agent_id:
        return True, "no agent to key the throttle on; notifying"

    try:
        recent_runs = store.list_runs(agent_id=str(agent_id), limit=10)
    except Exception:
        logger.exception(
            "failure-notification throttle: could not read run history for "
            "agent_id=%s; notifying rather than risking a swallowed failure",
            agent_id,
        )
        return True, "run-history read failed; notifying (fail open)"

    previous_run = _most_recent_other_run(recent_runs, exclude_run_id=run_id)
    if previous_run is None or previous_run.get("status") != "failed":
        return True, "new failure streak: the previous run for this agent did not fail"

    try:
        from agent.notification_store import get_notification_store

        recent = get_notification_store().list_notifications(
            agent_id=str(agent_id), level="error", limit=1
        )
    except Exception:
        logger.exception(
            "failure-notification throttle: could not read notification "
            "history for agent_id=%s; notifying (fail open)",
            agent_id,
        )
        return True, "notification-history read failed; notifying (fail open)"

    if not recent:
        return True, "no prior failure notification on record for this agent; notifying"

    last_at = _parse_notification_instant(recent[0].get("created_at"))
    if last_at is None:
        return True, "prior failure notification timestamp unreadable; notifying (fail open)"

    elapsed = datetime.now(UTC) - last_at
    if elapsed >= _NOTIFICATION_THROTTLE_WINDOW:
        return True, "24h throttle window elapsed since the last failure notification"

    return False, (
        f"same agent still failing consecutively; last notified "
        f"{recent[0].get('created_at')}, next eligible at "
        f"{(last_at + _NOTIFICATION_THROTTLE_WINDOW).isoformat()}"
    )


#: Waits that block every later run of the agent until a person acts, and that
#: a person clears in seconds. They are worth saying out loud, because the
#: notification about them is the only one that will arrive for a day.
_BLOCKING_WAIT_REASONS: frozenset[str] = frozenset(
    {"login_required", "captcha_or_bot_challenge"}
)


def _wait_notification_body(body: str, *, reason: str) -> str:
    """The park's own words, plus what the silence after them will mean.

    The throttle keeps this to one notification per agent per day, which is
    what the user asked for. What the notification never said is that it is
    the only one coming: an expired login stops every scheduled run of that
    agent, each one parks, each park is suppressed, and someone reading
    "needs a login" has no way to know the agent stays dead until they act.
    Saying so costs no extra push.
    """
    if reason not in _BLOCKING_WAIT_REASONS:
        return body
    note = (
        "이 에이전트의 예약 실행은 처리하실 때까지 계속 멈춥니다. "
        "같은 사유로는 24시간 동안 다시 알리지 않습니다."
    )
    return f"{body}\n{note}" if body else note


def _wait_notification_gate(
    *, store: Any, agent_id: str | None, run_id: str
) -> tuple[bool, str]:
    """Should this parked run push, or is it a quiet repeat of a known streak.

    ``_wait_for_user_step``'s own dedup (see its call site) already covers a
    resume that immediately re-parks on the *same run*: it compares this
    park's reason against the checkpoint recorded the last time that exact
    run parked. What it cannot see is a *new* run landing on the same wait —
    which happens, because task metadata's ``active_checkpoint`` is only
    cleared when a run finishes through ``_finish_workflow_execution``, and a
    scheduled run that stalls waiting on a human instead gets abandoned by
    ``agent.scheduler._abandon_stalled_run``, which marks the run failed
    directly and never goes through that finish path. A schedule that keeps
    landing on the same stuck login prompt, getting abandoned an hour later,
    and firing again would notify on every single firing — the identical
    four-a-day flood the failure throttle above exists to stop, just
    triggered by a stuck human handoff instead of a stuck script.

    Keyed on the agent alone, the same as the failure gate and for the same
    reason: the specific wait reason (login vs. captcha vs. approval) is not
    the thing the user asked to be throttled, and treating every reason
    change as a fresh event would mean an agent that oscillates between two
    wait reasons never goes quiet at all.

    A run that *completes* — the previous run for this agent actually
    finished successfully — ends the streak the same way a success ends the
    failure streak: this returns "notify" immediately even if a wait
    notification went out minutes ago, because a fresh park right after a
    real success is new information, not a repeat.

    Same fail-open rule as the failure gate: a notification-history read
    failure notifies rather than risking a swallowed first park.
    """
    if not agent_id:
        return True, "no agent to key the throttle on; notifying"

    try:
        recent_runs = store.list_runs(agent_id=str(agent_id), limit=10)
    except Exception:
        logger.exception(
            "wait-notification throttle: could not read run history for "
            "agent_id=%s; notifying rather than risking a swallowed park",
            agent_id,
        )
        return True, "run-history read failed; notifying (fail open)"

    previous_run = _most_recent_other_run(recent_runs, exclude_run_id=run_id)
    if previous_run is not None and previous_run.get("status") == "completed":
        return True, "the previous run for this agent finished successfully; fresh stuck streak"

    try:
        from agent.notification_store import get_notification_store

        recent = get_notification_store().list_notifications(
            agent_id=str(agent_id), level="warning", limit=1
        )
    except Exception:
        logger.exception(
            "wait-notification throttle: could not read notification history "
            "for agent_id=%s; notifying (fail open)",
            agent_id,
        )
        return True, "notification-history read failed; notifying (fail open)"

    if not recent:
        return True, "no prior wait notification on record for this agent; notifying"

    last_at = _parse_notification_instant(recent[0].get("created_at"))
    if last_at is None:
        return True, "prior wait notification timestamp unreadable; notifying (fail open)"

    elapsed = datetime.now(UTC) - last_at
    if elapsed >= _NOTIFICATION_THROTTLE_WINDOW:
        return True, "24h throttle window elapsed since the last wait notification"

    return False, (
        f"same agent still stuck waiting on a human; last notified "
        f"{recent[0].get('created_at')}, next eligible at "
        f"{(last_at + _NOTIFICATION_THROTTLE_WINDOW).isoformat()}"
    )


def _record_suppressed_notification(
    *, store: Any, run_id: str, kind: str, agent_id: str | None, why: str
) -> None:
    """Make a suppressed notification discoverable without touching run state.

    The run's own status/result is already written by the time either notify
    function runs — that record must never depend on whether a push went
    out. This only appends an ``agent_events`` row (visible on the run's
    existing timeline, the same place every other step/lifecycle event
    lives) plus a log line, so an operator looking at *why the phone stayed
    quiet* has an answer instead of an unexplained gap. Failure here is
    swallowed by the caller's own best-effort ``try`` — a broken audit trail
    is a much smaller problem than a broken run.
    """
    logger.info(
        "notification throttled: kind=%s run_id=%s agent_id=%s reason=%s",
        kind,
        run_id,
        agent_id,
        why,
    )
    store.append_event(
        run_id=run_id,
        event_type="notification.suppressed",
        app_event={"kind": kind, "agent_id": agent_id, "reason": why},
    )


def _notify_waiting_for_user_best_effort(
    *,
    store: Any,
    task: dict[str, Any] | None,
    run_id: str,
    step: dict[str, Any],
    reason: str,
    prompt: str | None,
) -> None:
    """Tell the phone a run parked for a human, best-effort.

    The run is already parked by the time this runs — its only job left is
    to make sure someone finds out. A notification-store failure or a push
    failure must never turn into a further-stalled or failed run; the run
    stays parked either way, and the worst outcome here is the user finding
    out from the app instead of a push, which is exactly today's behavior.
    Losing this path entirely, silently, is the actual regression to guard
    against: without it a 3am run stuck on a login prompt sits invisible
    until someone opens the app and happens to notice a badge.

    Only called (see the call site in ``_wait_for_user_step``) when this is
    not an immediate resume-and-re-park loop on the very same run. Even so,
    a schedule that keeps landing on the same wait reason across separate
    runs would otherwise notify every time it fires — see
    ``_wait_notification_gate`` for why that happens and why it is throttled
    the same way the failure notifier is, to once per agent per day while
    the streak holds.
    """
    try:
        agent_id = task.get("assigned_agent_id") if isinstance(task, dict) else None

        should_notify, why = _wait_notification_gate(
            store=store, agent_id=agent_id, run_id=run_id
        )
        if not should_notify:
            _record_suppressed_notification(
                store=store,
                run_id=run_id,
                kind="waiting_for_user",
                agent_id=agent_id,
                why=why,
            )
            return

        from agent.notification_store import get_notification_store

        agent_name = None
        if agent_id:
            agent = store.get_agent(str(agent_id))
            if isinstance(agent, dict):
                agent_name = agent.get("name")
        task_title = (
            (task.get("title") if isinstance(task, dict) else None)
            or step.get("title")
            or "A task"
        )
        reason_label = _wait_reason_label(reason)
        title = f"{agent_name or 'An agent'} needs you: {reason_label}"
        body = _wait_notification_body(
            prompt or f"{task_title} is waiting for you.", reason=reason
        )

        notification = get_notification_store().create(
            title=title,
            body=body,
            level="warning",
            run_id=run_id,
            task_id=task.get("id") if isinstance(task, dict) else None,
            agent_id=agent_id,
        )
        _push_notification_best_effort(
            notification=notification,
            title=title,
            body=body,
            level="warning",
        )
    except Exception:
        logger.exception(
            "Waiting-for-user notification failed for run_id=%s step_id=%s; run stays parked",
            run_id,
            step.get("id"),
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
    if status == "failed":
        _notify_run_failed_best_effort(
            store=store,
            task=task,
            run_id=run_id,
            error=error,
        )


def _notify_run_failed_best_effort(
    *,
    store: Any,
    task: dict[str, Any] | None,
    run_id: str,
    error: dict[str, Any] | None,
) -> None:
    """Tell the phone a run ended in failure, best-effort.

    An unattended agent that dies at 3am and says nothing is the harm this
    exists to prevent: the schedule keeps firing, the results panel keeps
    filling with red, and nobody knows until someone opens the app days later.
    A parked run already notifies (`_notify_waiting_for_user_best_effort`); a
    failed one is at least as worth hearing about, because unlike a parked run
    it will never resume on its own.

    Strictly best-effort, for the same reason as the parked-run path: the run
    is already finished and recorded by the time this runs. A notification
    store failure or a dead FCM key must never change what the run says it
    did — the worst acceptable outcome is the user reading it in the app
    instead of on a lock screen.

    Throttled: while the same agent keeps failing consecutively this notifies
    at most once a day, so a script that has been failing the same way every
    six hours for a week doesn't turn every one of those runs into a push —
    see ``_failure_notification_gate`` for the exact rule (first failure
    always notifies, a success resets the streak) and why it's worth having
    at all (a flood of identical pushes trains people to swipe them away
    unread, which is exactly how a push that matters — a run genuinely stuck,
    a file gone missing — gets swiped away too).
    """
    try:
        agent_id = task.get("assigned_agent_id") if isinstance(task, dict) else None

        should_notify, why = _failure_notification_gate(
            store=store, agent_id=agent_id, run_id=run_id
        )
        if not should_notify:
            _record_suppressed_notification(
                store=store,
                run_id=run_id,
                kind="failed",
                agent_id=agent_id,
                why=why,
            )
            return

        from agent.notification_store import get_notification_store

        agent_name = None
        if agent_id:
            agent = store.get_agent(str(agent_id))
            if isinstance(agent, dict):
                agent_name = agent.get("name")
        task_title = (task.get("title") if isinstance(task, dict) else None) or "A task"

        title = f"{agent_name or 'An agent'} run failed"
        body = "\n".join(
            [f"{task_title}: {_run_failure_reason(error)}", *_failed_step_lines(store, run_id)]
        )

        notification = get_notification_store().create(
            title=title,
            body=body,
            level="error",
            run_id=run_id,
            task_id=task.get("id") if isinstance(task, dict) else None,
            agent_id=agent_id,
        )
        _push_notification_best_effort(
            notification=notification,
            title=title,
            body=body,
            level="error",
        )
    except Exception:
        logger.exception(
            "Run-failed notification failed for run_id=%s; the run is still failed",
            run_id,
        )


def _run_failure_reason(error: dict[str, Any] | None) -> str:
    """One line saying why, for a notification body.

    "The run failed" with no reason is not actionable at 3am — the whole point
    of the message is that the reader can decide whether to get up.
    """
    candidates: list[Any] = []
    if isinstance(error, dict):
        candidates.append(error.get("message"))
        nested = error.get("error")
        if isinstance(nested, dict):
            candidates.append(nested.get("message"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return _truncate_workflow_evidence(candidate.strip(), 500)
    return "No reason was recorded."


def _failed_step_lines(store: Any, run_id: str) -> list[str]:
    """Name the steps that failed, with whatever evidence they kept.

    For a shell step that is the exit code and the tail of stderr, which is
    the difference between "the nightly device cycle failed" and "the phone
    was offline".
    """
    try:
        run = store.get_run(run_id)
        task_id = run.get("task_id") if isinstance(run, dict) else None
        if not isinstance(task_id, str) or not task_id:
            return []
        lines: list[str] = []
        for step in _steps_for_run(store, task_id, run_id):
            if step.get("status") != "failed":
                continue
            evidence = _most_telling_evidence_line(
                _workflow_step_output_summary(dict(step.get("output") or {}))
            )
            detail = f" — {evidence}" if evidence else ""
            lines.append(f"Failed step: {step.get('title') or step.get('id')}{detail}")
        return lines[:3]
    except Exception:
        logger.exception("could not summarize failed steps for run %s", run_id)
        return []


def _most_telling_evidence_line(summary: list[str]) -> str:
    """Pick the one line from a step's evidence worth putting in a push.

    A stderr tail ("adb: device offline") tells the reader whether to get out
    of bed; "shell.status: failed" does not.
    """
    for marker in (".stderr:", ".error:", ".exit_code:", ".message:"):
        for line in summary:
            if marker in line:
                return _truncate_workflow_evidence(line.strip(), 300)
    return _truncate_workflow_evidence(summary[0].strip(), 300) if summary else ""


def _workflow_step_message(
    step: dict[str, Any],
    launch_message: str,
    *,
    previous_steps: list[dict[str, Any]] | None = None,
    matched_memories: list[dict[str, Any]] | None = None,
    cli_agent_source_path: str | None = None,
) -> str:
    step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
    instruction = _workflow_text(
        step_input.get("instruction") or step_input.get("description")
    )
    observation = _workflow_text(step_input.get("observation"))
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
    if cli_agent_source_path:
        # The step's stored instruction is a reference stub, and on an agent
        # imported before this ran by reference it is a stale copy of the file.
        # Either way it is not what executes: the live file is already the
        # session's agent definition. Replaying it here would hand the model a
        # second, older set of instructions contradicting the first.
        lines.append(
            "- instruction: defined by the Claude Code agent file "
            f"{cli_agent_source_path}, loaded for this run and already in force "
            "as this session's agent definition"
        )
    elif instruction:
        lines.append(f"- instruction: {instruction}")
    if observation:
        lines.append(f"- observation: {observation}")
    # memory_read is a query, not prose to relay verbatim: it was already
    # resolved to the agent memories it matches (see ``_memories_for_step``).
    # Nothing here means nothing matched — say nothing rather than inventing
    # a section, and never paste the raw selector text into the prompt.
    if matched_memories:
        lines.append("- relevant memories:")
        for memory in matched_memories:
            content = _workflow_text(memory.get("content"))
            if content:
                lines.append(f"  - {content}")
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


#: Memories are authored as read/write over "what happened", but sentences
#: ("check the request id and outcome of prior installs") rarely reappear
#: verbatim in memory content. Tokenizing into keywords and matching any of
#: them is a predictable, ranking-free stand-in for full-text search — not a
#: relevance engine, just "does this memory look related".
_MEMORY_KEYWORD_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_MEMORY_READ_MATCH_LIMIT = 5


def _memory_read_keywords(query: str) -> list[str]:
    if not query:
        return []
    return [
        token.lower()
        for token in _MEMORY_KEYWORD_PATTERN.findall(query)
        if len(token) >= 2
    ]


def _match_agent_memories(
    memories: list[dict[str, Any]],
    query: str,
    *,
    limit: int = _MEMORY_READ_MATCH_LIMIT,
) -> list[dict[str, Any]]:
    keywords = _memory_read_keywords(query)
    if not keywords:
        return []
    matches: list[dict[str, Any]] = []
    for memory in memories:
        content = str(memory.get("content") or "").lower()
        if not content:
            continue
        if any(keyword in content for keyword in keywords):
            matches.append(memory)
        if len(matches) >= limit:
            break
    return matches


def _step_agent_id(task: dict[str, Any]) -> str | None:
    agent_id = task.get("assigned_agent_id")
    return agent_id if isinstance(agent_id, str) and agent_id else None


def _memories_for_step(
    store: Any,
    agent_id: str | None,
    step_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve ``memory_read`` to the agent memories it selects.

    ``memory_read`` is a query the author writes, not prose meant for the
    model verbatim — the whole point of wiring it is that the step's prompt
    carries the matched memories instead of the query text.
    """
    if not agent_id:
        return []
    query = _workflow_jsonish(
        step_input.get("memory_read") or step_input.get("memoryRead")
    )
    if not query:
        return []
    memories = store.list_memories(agent_id, limit=100) or []
    return _match_agent_memories(memories, query)


def _remember_step_result(
    store: Any,
    *,
    agent_id: str | None,
    run_id: str,
    step_input: dict[str, Any],
    result_text: str | None,
) -> None:
    """Persist a step's result as an agent memory when ``memory_write`` asks.

    What gets stored is the step's real result, not a rule-based summary of
    it — content this repo has no business inventing stays uninvented, so a
    step that produced nothing writes nothing rather than a fabricated note.
    Storage is best-effort: the step's actual work already succeeded by the
    time this runs, so a memory-store failure is logged and swallowed rather
    than turned into a failed step.
    """
    if not agent_id:
        return
    memory_write = _workflow_jsonish(
        step_input.get("memory_write") or step_input.get("memoryWrite")
    )
    if not memory_write:
        return
    if not result_text or not result_text.strip():
        return
    content = _truncate_workflow_evidence(result_text.strip(), 4000)
    try:
        store.add_memory(
            agent_id=agent_id,
            content=content,
            source_run_id=run_id,
            source_event_type="workflow_step",
        )
    except Exception:
        logger.exception(
            "failed to persist workflow step memory agent_id=%s run_id=%s",
            agent_id,
            run_id,
        )


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
    if status == "failed":
        # Same reasoning as the workflow path: an agent without a flow still
        # runs unattended on a schedule, and still has nobody watching it fail.
        _notify_run_failed_best_effort(
            store=store,
            task=store.get_task(task_id),
            run_id=run_id,
            error=error,
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


def _is_detected_mcp_capability(capability: dict[str, Any], name: str) -> bool:
    """An mcp_server catalog row for ``name`` that this machine really has.

    `_demote_stale_mcp_rows` leaves rows for servers that no longer exist in
    the catalog on purpose (existing task links must keep resolving), marked
    `status='unverified'` and `metadata.detected=False`. Those rows must not be
    picked up as if they were live.
    """
    if capability.get("type") != "mcp_server" or capability.get("name") != name:
        return False
    if str(capability.get("status") or "") != "available":
        return False
    metadata = capability.get("metadata")
    return not isinstance(metadata, dict) or metadata.get("detected") is not False


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
    # These two used to match the hardcoded github/gmail catalog rows, which
    # existed on every machine and on none: C5 stopped writing them and
    # downgraded the ones already in the database to `unverified`. The branches
    # stay, because a user who really does configure a github MCP server should
    # get it — but they now require a row the detection pass actually found, so
    # a leftover row can no longer put "this task will use gmail" on a task
    # whose machine has no gmail server.
    if any(token in text for token in ("github", "issue", "pr", "pull request")):
        add_by(lambda cap: _is_detected_mcp_capability(cap, "github"))
    if any(token in text for token in ("gmail", "email", "mail")):
        add_by(lambda cap: _is_detected_mcp_capability(cap, "gmail"))
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
                    "notify": workflow_step.get("notify") or {},
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
    source_path = find_cli_agent_source_path(agent_id)
    if source_path:
        # A file-backed agent's real prompt arrives as the session's agent
        # definition, read from the file at execution time. Whatever sits in
        # the stored system_prompt is a pointer at best and, for an agent
        # imported before this ran by reference, a stale copy — either way it
        # must not be pasted into the launch message as if it were the
        # instructions. Only the mapping is read here, never the file: planning
        # must not fail because a source moved, and a plan is not a run.
        agent = {
            **agent,
            "system_prompt": cli_agent_reference_prompt(
                str(agent.get("name") or agent_id), source_path
            ),
        }
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
