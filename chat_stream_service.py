"""Streaming service for websocket chat turns."""

import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from chat_event_utils import extract_assistant_text, format_tool_result_content
from llm_session import LlmSession
from claude_usage import fetch_claude_usage_snapshot, merge_usage_for_display
from config import get_config
from database import get_usage_db

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """Mutable state for a single chat turn."""

    full_response_chunks: list[str] = field(default_factory=list)
    fallback_response: str = ""
    seen_tool_use_ids: set[str] = field(default_factory=set)
    turn_completed: bool = False


async def _emit_tool_use(
    websocket: WebSocket,
    state: TurnState,
    tool_id: Any,
    tool_name: Any,
    tool_input: Any,
) -> None:
    """Emit tool_use event to websocket, deduplicating by tool_id."""
    resolved_id = tool_id if isinstance(tool_id, str) else None
    if resolved_id is not None:
        if resolved_id in state.seen_tool_use_ids:
            return
        state.seen_tool_use_ids.add(resolved_id)

    await websocket.send_json(
        {
            "type": "tool_use",
            "id": resolved_id,
            "name": tool_name,
            "input": tool_input if isinstance(tool_input, dict) else {},
        }
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
    """Handle assistant event - extract tool uses and fallback response."""
    message_payload = event.get("message", {})
    if not isinstance(message_payload, dict):
        return

    blocks = message_payload.get("content", [])
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            await _emit_tool_use(
                websocket,
                state,
                block.get("id"),
                block.get("name"),
                block.get("input"),
            )

    if not state.full_response_chunks and not state.fallback_response:
        state.fallback_response = extract_assistant_text(message_payload)


async def _handle_user_event(
    websocket: WebSocket,
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

        await websocket.send_json(
            {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id"),
                "is_error": bool(block.get("is_error", False)),
                "content": format_tool_result_content(block.get("content")),
            }
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

    await websocket.send_json(
        {
            "type": "turn_metrics",
            "duration_ms": event.get("duration_ms"),
            "duration_api_ms": event.get("duration_api_ms"),
            "num_turns": event.get("num_turns"),
            "total_cost_usd": usage_info["total_cost_usd"],
            "usage": usage_info["usage_dict"],
            "model_usage": usage_info["model_usage_dict"],
        }
    )

    try:
        usage_db = get_usage_db()
        usage_db.record_turn(
            project_name=project_name,
            cost_usd=usage_info["total_cost_usd"],
            input_tokens=usage_info["input_tokens"],
            output_tokens=usage_info["output_tokens"],
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
                "event": {
                    "type": "system",
                    "subtype": "status",
                    "status": f"Usage summary update failed: {exc}",
                },
            }
        )


async def _handle_control_request(
    websocket: WebSocket,
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
        await websocket.send_json({"type": "claude_event", "event": event})
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

        denials = [
            {
                "request_id": request_id,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "input": tool_input if isinstance(tool_input, dict) else {},
            }
        ]
        await websocket.send_json(
            {
                "type": "permission_required",
                "denials": denials,
                "request_id": request_id,
                "message": (
                    f"Tool '{tool_name}' requires approval to continue."
                    if isinstance(tool_name, str) and tool_name
                    else "A tool requires approval to continue."
                ),
            }
        )
        return False

    await websocket.send_json({"type": "claude_event", "event": event})
    return None


async def _handle_error_event(
    websocket: WebSocket,
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
    await websocket.send_json({"type": "error", "message": error_message})


async def _handle_output_event(
    websocket: WebSocket,
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


async def stream_claude_turn(
    websocket: WebSocket,
    session: LlmSession,
    project_name: str,
    user_message: str | None = None,
    retry_from_permission: bool = False,
    deny_from_permission_message: str | None = None,
) -> bool:
    """Stream one Claude turn and forward events to websocket client."""
    logger.info(
        "[chat_stream] project=%s turn_start retry=%s deny=%s",
        project_name,
        retry_from_permission,
        deny_from_permission_message is not None,
    )

    state = TurnState()

    # Select event stream based on mode
    if deny_from_permission_message is not None:
        event_stream = session.deny_pending_permissions(message=deny_from_permission_message)
    elif retry_from_permission:
        event_stream = session.approve_pending_permissions_and_retry()
    else:
        if user_message is None:
            raise ValueError("user_message is required when retry_from_permission is False")
        event_stream = session.send_message(user_message)

    # Process events
    async for event in event_stream:
        event_type = event.get("type")

        if event_type == "stream_event":
            await _handle_stream_event(websocket, state, event)
            continue

        if event_type == "assistant":
            await _handle_assistant_event(websocket, state, event)
            continue

        if event_type == "user":
            await _handle_user_event(websocket, event)
            continue

        if event_type == "result":
            await _handle_result_event(websocket, state, event, project_name)
            continue

        if event_type == "control_request":
            result = await _handle_control_request(websocket, event, project_name)
            if result is False:
                return False
            continue

        if event_type == "error":
            await _handle_error_event(websocket, event, project_name)
            continue

        if event_type == "output":
            await _handle_output_event(websocket, event, project_name)
            continue

        # Pass through unknown events
        await websocket.send_json({"type": "claude_event", "event": event})

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
    return True
