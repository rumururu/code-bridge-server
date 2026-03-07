"""Streaming service for websocket chat turns."""

from typing import Any

from fastapi import WebSocket

from chat_event_utils import extract_assistant_text, format_tool_result_content
from claude_session import ClaudeSession
from claude_usage import fetch_claude_usage_snapshot, merge_usage_for_display
from config import get_config
from database import get_usage_db


async def stream_claude_turn(
    websocket: WebSocket,
    session: ClaudeSession,
    project_name: str,
    user_message: str | None = None,
    retry_from_permission: bool = False,
    deny_from_permission_message: str | None = None,
) -> bool:
    """Stream one Claude turn and forward events to websocket client."""
    print(
        "[chat_stream] "
        f"project={project_name} turn_start retry={retry_from_permission} "
        f"deny={deny_from_permission_message is not None}"
    )
    full_response_chunks: list[str] = []
    fallback_response = ""
    seen_tool_use_ids: set[str] = set()
    turn_completed = False

    async def emit_tool_use(
        tool_id: Any,
        tool_name: Any,
        tool_input: Any,
    ) -> None:
        resolved_id = tool_id if isinstance(tool_id, str) else None
        if resolved_id is not None:
            if resolved_id in seen_tool_use_ids:
                return
            seen_tool_use_ids.add(resolved_id)

        await websocket.send_json(
            {
                "type": "tool_use",
                "id": resolved_id,
                "name": tool_name,
                "input": tool_input if isinstance(tool_input, dict) else {},
            }
        )

    if deny_from_permission_message is not None:
        event_stream = session.deny_pending_permissions(message=deny_from_permission_message)
    elif retry_from_permission:
        event_stream = session.approve_pending_permissions_and_retry()
    else:
        if user_message is None:
            raise ValueError("user_message is required when retry_from_permission is False")
        event_stream = session.send_message(user_message)

    async for event in event_stream:
        event_type = event.get("type")

        if event_type == "stream_event":
            stream_event = event.get("event", {})
            if not isinstance(stream_event, dict):
                continue

            stream_type = stream_event.get("type")
            if stream_type == "content_block_start":
                content_block = stream_event.get("content_block", {})
                if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                    await emit_tool_use(
                        content_block.get("id"),
                        content_block.get("name"),
                        content_block.get("input"),
                    )
                continue

            if stream_type == "content_block_delta":
                delta = stream_event.get("delta", {})
                if not isinstance(delta, dict):
                    continue

                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if isinstance(text, str) and text:
                        full_response_chunks.append(text)
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
                continue

            continue

        if event_type == "assistant":
            message_payload = event.get("message", {})
            if isinstance(message_payload, dict):
                blocks = message_payload.get("content", [])
                if isinstance(blocks, list):
                    for block in blocks:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        await emit_tool_use(
                            block.get("id"),
                            block.get("name"),
                            block.get("input"),
                        )

                if not full_response_chunks and not fallback_response:
                    fallback_response = extract_assistant_text(message_payload)
            continue

        if event_type == "user":
            message_payload = event.get("message", {})
            if not isinstance(message_payload, dict):
                continue

            blocks = message_payload.get("content", [])
            if not isinstance(blocks, list):
                continue

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
            continue

        if event_type == "result":
            turn_completed = True
            print(f"[chat_stream] project={project_name} event=result")
            if not fallback_response:
                result_text = event.get("result")
                if isinstance(result_text, str):
                    fallback_response = result_text

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

            await websocket.send_json(
                {
                    "type": "turn_metrics",
                    "duration_ms": event.get("duration_ms"),
                    "duration_api_ms": event.get("duration_api_ms"),
                    "num_turns": event.get("num_turns"),
                    "total_cost_usd": total_cost_usd,
                    "usage": usage_dict,
                    "model_usage": model_usage_dict,
                }
            )

            try:
                usage_db = get_usage_db()
                usage_db.record_turn(
                    project_name=project_name,
                    cost_usd=total_cost_usd,
                    input_tokens=input_tokens_int,
                    output_tokens=output_tokens_int,
                )
                config = get_config()
                weekly_summary = usage_db.get_weekly_summary(
                    budget_usd=config.weekly_budget_usd,
                    window_days=config.usage_window_days,
                )
                claude_snapshot = await fetch_claude_usage_snapshot()
                merged_usage = merge_usage_for_display(weekly_summary, claude_snapshot)
                await websocket.send_json({"type": "weekly_usage", **merged_usage})
            except Exception as exc:
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
            continue

        if event_type == "control_request":
            request = event.get("request", {})
            if not isinstance(request, dict):
                await websocket.send_json({"type": "claude_event", "event": event})
                continue

            if request.get("subtype") == "can_use_tool":
                tool_name = request.get("tool_name")
                print(
                    "[chat_stream] "
                    f"project={project_name} permission_required tool={tool_name}"
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
            continue

        if event_type == "error":
            error_payload = event.get("error")
            if isinstance(error_payload, dict):
                error_message = str(error_payload.get("message", "Unknown error"))
            else:
                error_message = str(error_payload or "Unknown error")
            print(
                "[chat_stream] "
                f"project={project_name} event=error message={error_message[:200]}"
            )
            await websocket.send_json({"type": "error", "message": error_message})
            continue

        if event_type == "output":
            text = event.get("text")
            if isinstance(text, str) and text:
                print(
                    "[chat_stream] "
                    f"project={project_name} event=status message={text[:200]}"
                )
                await websocket.send_json({"type": "status", "message": text})
            continue

        await websocket.send_json({"type": "claude_event", "event": event})

    if not turn_completed:
        print(f"[chat_stream] project={project_name} turn_end completed=False")
        return False

    final_response = "".join(full_response_chunks).strip()
    if not final_response:
        final_response = fallback_response.strip()

    print(
        "[chat_stream] "
        f"project={project_name} turn_end completed=True response_len={len(final_response)}"
    )
    await websocket.send_json({"type": "complete", "content": final_response})
    return True
