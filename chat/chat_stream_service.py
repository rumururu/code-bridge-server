"""Streaming service for websocket chat turns."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

from .chat_event_utils import extract_assistant_text, format_tool_result_content
from approvals.approval_service import request_approval_for_operation
from agent.agent_store import get_agent_store
from llm.llm_session import LlmSession
from llm.claude_usage import fetch_claude_usage_snapshot, merge_usage_for_display
from core.config import get_config
from core.database import get_usage_db
from projects.project_manager import get_project_manager

logger = logging.getLogger(__name__)
APP_EVENT_SCHEMA_VERSION = 1
PROVIDER_EVENT_SCHEMA_VERSION = 1


def _build_marionette_context(project_name: str) -> str | None:
    """Build Marionette context reminder if Flutter app is running.

    Returns a system-reminder string with VM Service URI and usage instructions,
    or None if no app is running.
    """
    pm = get_project_manager()
    status = pm.get_device_run_status(project_name)

    if not status.get("running"):
        return None

    vm_uri = status.get("vm_service_uri")
    if not vm_uri:
        return None

    device_id = status.get("device_id", "unknown")

    return f"""<system-reminder>
## Flutter App Runtime Context

A Flutter app is currently running and you can interact with it using Marionette MCP.

**Device:** {device_id}
**VM Service URI:** {vm_uri}

### Available Commands:

1. **Connect to app** (required before other commands):
   ```
   mcp__marionette__connect(uri: "{vm_uri}")
   ```

2. **Hot Reload** (apply code changes):
   ```
   mcp__marionette__hot_reload()
   ```

3. **Get interactive elements** (find tappable widgets):
   ```
   mcp__marionette__get_interactive_elements()
   ```

4. **Tap element** (by key, text, or coordinates):
   ```
   mcp__marionette__tap(text: "Button Text")
   mcp__marionette__tap(key: "my_button_key")
   ```

5. **Enter text** (requires widget key):
   ```
   mcp__marionette__enter_text(key: "email_field", input: "test@example.com")
   ```

6. **Take screenshot**:
   ```
   mcp__marionette__take_screenshots()
   ```

**Workflow:** connect → get_interactive_elements → tap/enter_text → take_screenshots
</system-reminder>"""


@dataclass
class TurnState:
    """Mutable state for a single chat turn."""

    provider_id: str = "unknown"
    provider: str = "unknown"
    session_id: str | None = None
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sequence: int = 0
    provider_sequence: int = 0
    full_response_chunks: list[str] = field(default_factory=list)
    fallback_response: str = ""
    seen_tool_use_ids: set[str] = field(default_factory=set)
    turn_completed: bool = False

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def next_provider_sequence(self) -> int:
        self.provider_sequence += 1
        return self.provider_sequence


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _provider_id_for_session(session: LlmSession) -> str:
    provider_id = getattr(session, "provider_id", None)
    return provider_id if isinstance(provider_id, str) and provider_id else "unknown"


def _session_id_from_event(event: dict[str, Any]) -> str | None:
    session_id = event.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id

    raw_event = event.get("raw_event")
    if isinstance(raw_event, dict):
        raw_session_id = raw_event.get("session_id")
        if isinstance(raw_session_id, str) and raw_session_id.strip():
            return raw_session_id

    return None


def _raw_event_from(event: dict[str, Any]) -> dict[str, Any]:
    raw_event = event.get("raw_event")
    if isinstance(raw_event, dict):
        return raw_event

    nested_event = event.get("event")
    if event.get("type") in {"provider_event", "codex_event", "gemini_event"} and isinstance(
        nested_event,
        dict,
    ):
        return nested_event

    return event


def _normalized_event_from(event: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: value for key, value in event.items() if key != "raw_event"}
    if normalized.get("type") == "provider_event" and isinstance(
        normalized.get("normalized"),
        dict,
    ):
        nested = normalized["normalized"]
        return {key: value for key, value in nested.items() if key != "raw_event"}
    return normalized


def _approval_operation_for_tool(tool_name: Any) -> str:
    """Map provider-native tool names to Code Bridge policy operations."""
    normalized = str(tool_name or "").strip().lower()
    if normalized in {"bash", "shell", "terminal", "run_command"}:
        return "process.terminal"
    if normalized in {"edit", "multiedit", "write", "notebookedit"}:
        return "file.write"
    if normalized in {"git", "git_commit", "git_push"}:
        return "git.commit"
    return "provider.tool"


async def _emit_app_event(
    websocket: WebSocket,
    state: TurnState,
    name: str,
    *,
    title: str,
    detail: str | None = None,
    level: str = "info",
    data: dict[str, Any] | None = None,
    raw_event: dict[str, Any] | None = None,
) -> None:
    """Emit a compact, app-oriented event while keeping legacy events intact."""
    payload: dict[str, Any] = {
        "type": "app_event",
        "schema_version": APP_EVENT_SCHEMA_VERSION,
        "event": name,
        "provider_id": state.provider_id,
        "provider": state.provider,
        "session_id": state.session_id,
        "turn_id": state.turn_id,
        "sequence": state.next_sequence(),
        "timestamp": _timestamp(),
        "title": title,
        "level": level,
    }
    if detail:
        payload["detail"] = detail
    if data:
        payload["data"] = data
    if raw_event is not None:
        payload["raw_event"] = raw_event
    await websocket.send_json(payload)


async def _emit_provider_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
) -> None:
    """Emit a provider-neutral raw provider event envelope."""
    provider_id = event.get("provider_id")
    if isinstance(provider_id, str) and provider_id.strip():
        state.provider_id = provider_id
    provider = event.get("provider")
    if isinstance(provider, str) and provider.strip():
        state.provider = provider
    elif isinstance(provider_id, str) and provider_id.strip():
        state.provider = provider_id

    session_id = _session_id_from_event(event)
    if session_id is not None:
        state.session_id = session_id

    await websocket.send_json(
        {
            "type": "provider_event",
            "schema_version": PROVIDER_EVENT_SCHEMA_VERSION,
            "provider_id": state.provider_id,
            "provider": state.provider,
            "session_id": state.session_id,
            "turn_id": state.turn_id,
            "sequence": state.next_provider_sequence(),
            "timestamp": _timestamp(),
            "event": _raw_event_from(event),
            "normalized": _normalized_event_from(event),
        }
    )


async def _emit_legacy_provider_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
) -> None:
    """Emit legacy provider passthrough for clients still listening to claude_event."""
    await websocket.send_json(
        {
            "type": "claude_event",
            "provider_id": state.provider_id,
            "provider": state.provider,
            "session_id": state.session_id,
            "turn_id": state.turn_id,
            "event": _raw_event_from(event),
        }
    )


async def _emit_tool_use(
    websocket: WebSocket,
    state: TurnState,
    tool_id: Any,
    tool_name: Any,
    tool_input: Any,
    raw_event: dict[str, Any] | None = None,
) -> None:
    """Emit tool_use event to websocket, deduplicating by tool_id."""
    resolved_id = tool_id if isinstance(tool_id, str) else None
    if resolved_id is not None:
        if resolved_id in state.seen_tool_use_ids:
            return
        state.seen_tool_use_ids.add(resolved_id)

    payload: dict[str, Any] = {
        "type": "tool_use",
        "id": resolved_id,
        "name": tool_name,
        "input": tool_input if isinstance(tool_input, dict) else {},
    }
    if raw_event is not None:
        payload["raw_event"] = raw_event
    await websocket.send_json(payload)
    tool_label = str(tool_name).strip() if tool_name else "tool"
    await _emit_app_event(
        websocket,
        state,
        "tool.started",
        title=f"$ {tool_label}",
        detail=None,
        data={
            "tool_id": resolved_id,
            "tool_name": tool_name,
            "input": tool_input if isinstance(tool_input, dict) else {},
        },
        raw_event=raw_event,
    )


async def _handle_stream_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
) -> None:
    """Handle stream_event type - content deltas and tool use starts."""
    stream_event = event.get("event", {})
    if not isinstance(stream_event, dict):
        return

    stream_type = stream_event.get("type")

    if stream_type == "content_block_start":
        content_block = stream_event.get("content_block", {})
        if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
            await _emit_tool_use(
                websocket,
                state,
                content_block.get("id"),
                content_block.get("name"),
                content_block.get("input"),
                raw_event=_raw_event_from(event),
            )
        return

    if stream_type == "content_block_delta":
        delta = stream_event.get("delta", {})
        if not isinstance(delta, dict):
            return

        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text", "")
            if isinstance(text, str) and text:
                state.full_response_chunks.append(text)
                await websocket.send_json({"type": "stream", "content": text})
        elif delta_type == "input_json_delta":
            partial_json = delta.get("partial_json", "")
            if isinstance(partial_json, str) and partial_json:
                await websocket.send_json(
                    {
                        "type": "tool_input_delta",
                        "content": partial_json,
                        "index": stream_event.get("index"),
                    }
                )


async def _handle_assistant_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
) -> None:
    """Handle assistant event - extract tool events and fallback response."""
    message_payload = event.get("message", {})
    if not isinstance(message_payload, dict):
        return

    blocks = message_payload.get("content", [])
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                await _emit_tool_result(
                    websocket,
                    state,
                    block,
                    raw_event=_raw_event_from(event),
                )
                continue
            if block_type != "tool_use":
                continue
            await _emit_tool_use(
                websocket,
                state,
                block.get("id"),
                block.get("name"),
                block.get("input"),
                raw_event=_raw_event_from(event),
            )

    if not state.full_response_chunks and not state.fallback_response:
        state.fallback_response = extract_assistant_text(message_payload)


async def _handle_user_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
) -> None:
    """Handle user event - forward tool results."""
    message_payload = event.get("message", {})
    if not isinstance(message_payload, dict):
        return

    blocks = message_payload.get("content", [])
    if not isinstance(blocks, list):
        return

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        await _emit_tool_result(websocket, state, block, raw_event=_raw_event_from(event))


async def _emit_tool_result(
    websocket: WebSocket,
    state: TurnState,
    block: dict[str, Any],
    *,
    raw_event: dict[str, Any] | None = None,
) -> None:
    """Emit a normalized tool result regardless of provider event role."""
    is_error = bool(block.get("is_error", False))
    content = format_tool_result_content(block.get("content"))
    payload: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": block.get("tool_use_id"),
        "is_error": is_error,
        "content": content,
    }
    if raw_event is not None:
        payload["raw_event"] = raw_event
    await websocket.send_json(payload)
    await _emit_app_event(
        websocket,
        state,
        "tool.completed",
        title="tool failed" if is_error else "tool done",
        detail=content,
        level="error" if is_error else "info",
        data={
            "tool_use_id": block.get("tool_use_id"),
            "is_error": is_error,
        },
        raw_event=raw_event,
    )


def _extract_usage_from_result(event: dict[str, Any]) -> dict[str, Any]:
    """Extract usage metrics from result event."""
    usage = event.get("usage")
    usage_dict = usage if isinstance(usage, dict) else {}

    model_usage = event.get("modelUsage")
    if not isinstance(model_usage, dict):
        model_usage = event.get("model_usage")
    model_usage_dict = model_usage if isinstance(model_usage, dict) else {}

    total_cost = event.get("total_cost_usd")
    if isinstance(total_cost, (int, float)):
        total_cost_usd = float(total_cost)
    else:
        try:
            total_cost_usd = float(str(total_cost))
        except (TypeError, ValueError):
            total_cost_usd = 0.0

    input_tokens = usage_dict.get("input_tokens")
    output_tokens = usage_dict.get("output_tokens")
    try:
        input_tokens_int = int(input_tokens) if input_tokens is not None else 0
    except (TypeError, ValueError):
        input_tokens_int = 0
    try:
        output_tokens_int = int(output_tokens) if output_tokens is not None else 0
    except (TypeError, ValueError):
        output_tokens_int = 0

    return {
        "usage_dict": usage_dict,
        "model_usage_dict": model_usage_dict,
        "total_cost_usd": total_cost_usd,
        "input_tokens": input_tokens_int,
        "output_tokens": output_tokens_int,
    }


async def _handle_result_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
    project_name: str,
) -> None:
    """Handle result event - send metrics and update usage."""
    state.turn_completed = True
    logger.info("[chat_stream] project=%s event=result", project_name)

    if not state.fallback_response:
        result_text = event.get("result")
        if isinstance(result_text, str):
            state.fallback_response = result_text

    usage_info = _extract_usage_from_result(event)

    raw_event = _raw_event_from(event)
    await websocket.send_json(
        {
            "type": "turn_metrics",
            "duration_ms": event.get("duration_ms"),
            "duration_api_ms": event.get("duration_api_ms"),
            "num_turns": event.get("num_turns"),
            "total_cost_usd": usage_info["total_cost_usd"],
            "usage": usage_info["usage_dict"],
            "model_usage": usage_info["model_usage_dict"],
            "raw_event": raw_event,
        }
    )
    await _emit_app_event(
        websocket,
        state,
        "turn.metrics",
        title="usage",
        detail=(
            f"in {usage_info['input_tokens']} · out {usage_info['output_tokens']} · "
            f"${usage_info['total_cost_usd']:.4f}"
        ),
        data={
            "input_tokens": usage_info["input_tokens"],
            "output_tokens": usage_info["output_tokens"],
            "total_cost_usd": usage_info["total_cost_usd"],
        },
        raw_event=raw_event,
    )

    try:
        usage_db = get_usage_db()
        usage_db.record_turn(
            project_name=project_name,
            cost_usd=usage_info["total_cost_usd"],
            input_tokens=usage_info["input_tokens"],
            output_tokens=usage_info["output_tokens"],
        )
        agent_run_id = getattr(websocket, "agent_run_id", None)
        run = (
            get_agent_store().get_run(agent_run_id)
            if isinstance(agent_run_id, str) and agent_run_id
            else None
        )
        usage_db.record_event(
            source="chat",
            project_name=None if project_name == "__global__" else project_name,
            workspace_id=run.get("workspace_id") if run else None,
            task_id=run.get("task_id") if run else None,
            run_id=agent_run_id if isinstance(agent_run_id, str) else None,
            provider_id=state.provider_id,
            model=getattr(state, "model", None),
            native_session_id=state.session_id,
            turn_id=state.turn_id,
            duration_ms=(
                int(event.get("duration_ms"))
                if isinstance(event.get("duration_ms"), (int, float))
                else None
            ),
            input_tokens=usage_info["input_tokens"],
            output_tokens=usage_info["output_tokens"],
            cost_usd=usage_info["total_cost_usd"],
            raw_usage={
                "usage": usage_info["usage_dict"],
                "model_usage": usage_info["model_usage_dict"],
                "raw_event": raw_event,
            },
        )
        config = get_config()
        weekly_summary = usage_db.get_weekly_summary(
            budget_usd=config.weekly_budget_usd,
            window_days=config.usage_window_days,
        )
        claude_snapshot = await fetch_claude_usage_snapshot()
        merged_usage = merge_usage_for_display(weekly_summary, claude_snapshot)
        await websocket.send_json({"type": "weekly_usage", **merged_usage})
    except OSError as exc:
        await websocket.send_json(
            {
                "type": "claude_event",
                "provider_id": state.provider_id,
                "provider": state.provider,
                "session_id": state.session_id,
                "turn_id": state.turn_id,
                "event": {
                    "type": "system",
                    "subtype": "status",
                    "status": f"Usage summary update failed: {exc}",
                },
            }
        )


async def _handle_control_request(
    websocket: WebSocket,
    session: LlmSession,
    state: TurnState,
    event: dict[str, Any],
    project_name: str,
) -> bool | None:
    """Handle control_request event - permission prompts.

    Returns:
        False if permission is required (turn paused)
        None if event was handled but turn continues
    """
    request = event.get("request", {})
    if not isinstance(request, dict):
        await _emit_legacy_provider_event(websocket, state, event)
        return None

    if request.get("subtype") == "can_use_tool":
        tool_name = request.get("tool_name")
        logger.info(
            "[chat_stream] project=%s permission_required tool=%s",
            project_name,
            tool_name,
        )
        tool_input = request.get("input")
        request_id = event.get("request_id")
        tool_use_id = request.get("tool_use_id")
        approval_id = None
        approval_result: dict[str, Any] | None = None
        agent_run_id = getattr(websocket, "agent_run_id", None)
        if isinstance(agent_run_id, str) and agent_run_id.strip():
            approval_result = request_approval_for_operation(
                operation=_approval_operation_for_tool(tool_name),
                run_id=agent_run_id,
                actor={"type": "agent_session"},
                details={
                    "project_name": project_name,
                    "provider_id": state.provider_id,
                    "session_id": state.session_id,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "input": tool_input if isinstance(tool_input, dict) else {},
                    "provider_request_id": request_id,
                },
            )
            approval = (
                approval_result.get("approval") if isinstance(approval_result, dict) else None
            )
            approval_id = approval.get("id") if isinstance(approval, dict) else None

        if isinstance(approval_result, dict) and approval_result.get("allowed") is True:
            await _emit_app_event(
                websocket,
                state,
                "permission.auto_approved",
                title="permission auto-approved",
                detail=str(tool_name) if tool_name else None,
                data={
                    "request_id": request_id,
                    "policy": approval_result.get("policy"),
                },
                raw_event=_raw_event_from(event),
            )
            return await stream_claude_turn(
                websocket,
                session,
                project_name=project_name,
                retry_from_permission=True,
            )

        if isinstance(approval_result, dict) and approval_result.get("error"):
            policy = approval_result.get("policy")
            reason = (
                policy.get("reason")
                if isinstance(policy, dict) and isinstance(policy.get("reason"), str)
                else "Permission denied by policy."
            )
            await _emit_app_event(
                websocket,
                state,
                "permission.policy_denied",
                title="permission denied by policy",
                detail=str(tool_name) if tool_name else None,
                level="warning",
                data={
                    "request_id": request_id,
                    "policy": policy,
                    "reason": reason,
                },
                raw_event=_raw_event_from(event),
            )
            return await stream_claude_turn(
                websocket,
                session,
                project_name=project_name,
                deny_from_permission_message=reason,
            )

        denials = [
            {
                "request_id": request_id,
                "approval_id": approval_id,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "input": tool_input if isinstance(tool_input, dict) else {},
                "policy": approval_result.get("policy") if isinstance(approval_result, dict) else None,
                "desktop_only": bool(
                    approval_result.get("policy", {}).get("desktop_only")
                    if isinstance(approval_result, dict) and isinstance(approval_result.get("policy"), dict)
                    else False
                ),
            }
        ]
        policy = approval_result.get("policy") if isinstance(approval_result, dict) else None
        await websocket.send_json(
            {
                "type": "permission_required",
                "denials": denials,
                "request_id": request_id,
                "approval_id": approval_id,
                "policy": policy,
                "desktop_only": bool(policy.get("desktop_only")) if isinstance(policy, dict) else False,
                "message": (
                    f"Tool '{tool_name}' requires approval to continue."
                    if isinstance(tool_name, str) and tool_name
                    else "A tool requires approval to continue."
                ),
            }
        )
        await _emit_app_event(
            websocket,
            state,
            "permission.requested",
            title="permission required",
            detail=str(tool_name) if tool_name else None,
            level="warning",
            data={"denials": denials, "request_id": request_id},
            raw_event=_raw_event_from(event),
        )
        return False

    await _emit_legacy_provider_event(websocket, state, event)
    return None


async def _handle_error_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
    project_name: str,
) -> None:
    """Handle error event."""
    error_payload = event.get("error")
    if isinstance(error_payload, dict):
        error_message = str(error_payload.get("message", "Unknown error"))
    else:
        error_message = str(error_payload or "Unknown error")

    logger.warning(
        "[chat_stream] project=%s event=error message=%s",
        project_name,
        error_message[:200],
    )
    raw_event = _raw_event_from(event)
    await websocket.send_json(
        {"type": "error", "message": error_message, "raw_event": raw_event}
    )
    await _emit_app_event(
        websocket,
        state,
        "turn.failed",
        title="error",
        detail=error_message,
        level="error",
        raw_event=raw_event,
    )


async def _handle_output_event(
    websocket: WebSocket,
    state: TurnState,
    event: dict[str, Any],
    project_name: str,
) -> None:
    """Handle output event - status messages."""
    text = event.get("text")
    if isinstance(text, str) and text:
        logger.debug(
            "[chat_stream] project=%s event=status message=%s",
            project_name,
            text[:200],
        )
        await websocket.send_json({"type": "status", "message": text})
        await _emit_app_event(
            websocket,
            state,
            "turn.status",
            title=text,
            raw_event=_raw_event_from(event),
        )


async def stream_claude_turn(
    websocket: WebSocket,
    session: LlmSession,
    project_name: str,
    user_message: str | None = None,
    retry_from_permission: bool = False,
    deny_from_permission_message: str | None = None,
) -> bool:
    """Stream one Claude turn and forward events to websocket client."""
    logger.warning(
        "[chat_stream] BEGIN stream_claude_turn project=%s session_type=%s",
        project_name, type(session).__name__,
    )
    logger.info(
        "[chat_stream] project=%s turn_start retry=%s deny=%s",
        project_name,
        retry_from_permission,
        deny_from_permission_message is not None,
    )

    provider_id = _provider_id_for_session(session)
    session_id = getattr(session, "session_id", None)
    state = TurnState(
        provider_id=provider_id,
        provider=provider_id,
        session_id=session_id if isinstance(session_id, str) and session_id else None,
    )
    await _emit_app_event(
        websocket,
        state,
        "turn.started",
        title="turn started",
    )

    # Select event stream based on mode
    if deny_from_permission_message is not None:
        event_stream = session.deny_pending_permissions(message=deny_from_permission_message)
    elif retry_from_permission:
        event_stream = session.approve_pending_permissions_and_retry()
    else:
        if user_message is None:
            raise ValueError("user_message is required when retry_from_permission is False")

        # Inject Marionette context if Flutter app is running
        marionette_ctx = _build_marionette_context(project_name)
        if marionette_ctx:
            user_message = f"{user_message}\n\n{marionette_ctx}"

        event_stream = session.send_message(user_message)

    # Process events
    async for event in event_stream:
        if not isinstance(event, dict):
            continue
        await _emit_provider_event(websocket, state, event)
        event_type = event.get("type")

        if event_type == "stream_event":
            await _handle_stream_event(websocket, state, event)
            continue

        if event_type == "assistant":
            await _handle_assistant_event(websocket, state, event)
            continue

        if event_type == "user":
            await _handle_user_event(websocket, state, event)
            continue

        if event_type == "result":
            await _handle_result_event(websocket, state, event, project_name)
            continue

        if event_type == "control_request":
            result = await _handle_control_request(websocket, session, state, event, project_name)
            if result is False:
                return False
            if result is True:
                return True
            continue

        if event_type == "error":
            await _handle_error_event(websocket, state, event, project_name)
            continue

        if event_type == "output":
            await _handle_output_event(websocket, state, event, project_name)
            continue

        if event_type == "provider_event":
            await _emit_legacy_provider_event(websocket, state, event)
            continue

        # Pass through unknown events
        await _emit_legacy_provider_event(websocket, state, event)

    # Finalize turn
    if not state.turn_completed:
        logger.info("[chat_stream] project=%s turn_end completed=False", project_name)
        return False

    final_response = "".join(state.full_response_chunks).strip()
    if not final_response:
        final_response = state.fallback_response.strip()

    logger.info(
        "[chat_stream] project=%s turn_end completed=True response_len=%d",
        project_name,
        len(final_response),
    )
    await websocket.send_json({"type": "complete", "content": final_response})
    await _emit_app_event(
        websocket,
        state,
        "turn.completed",
        title="done",
        detail=f"{len(final_response)} chars",
    )
    return True
