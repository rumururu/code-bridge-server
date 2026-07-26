"""Convert claude-agent-sdk message objects back to the server's event dicts.

The chat stream contract (``chat_stream_service`` dispatch, ``LlmSession``
docstring) is a stream of plain dicts keyed by ``type``: ``assistant``,
``user``, ``result``, ``stream_event``, ``control_request``, ``error``,
``output``. That contract predates the SDK and is what the app parses, so the
SDK's typed messages are translated back into it here rather than rippling a
new shape through every consumer.

Anything unrecognised is passed through as a best-effort dict — the dispatch
in ``chat_stream_service`` forwards unknown events to the client instead of
dropping them, so a new SDK message type degrades to "shown but not specially
handled" rather than disappearing.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

__all__ = ["block_to_dict", "message_to_event", "session_id_of"]


def block_to_dict(block: Any) -> dict[str, Any]:
    """Render one content block in the wire shape the app already parses."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        payload: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
        }
        if block.content is not None:
            payload["content"] = block.content
        if block.is_error is not None:
            payload["is_error"] = block.is_error
        return payload
    if isinstance(block, dict):
        return block
    if is_dataclass(block):
        return asdict(block)
    return {"type": "text", "text": str(block)}


def _content_to_list(content: Any) -> list[dict[str, Any]] | str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [block_to_dict(block) for block in content]
    return []


def session_id_of(message: Any) -> str | None:
    """Session id carried by an SDK message, if any.

    The CLI used to report this on every raw line; the SDK hangs it off the
    message objects instead, and it is what ``--resume`` needs later.
    """
    session_id = getattr(message, "session_id", None)
    if isinstance(session_id, str) and session_id.strip():
        return session_id
    if isinstance(message, SystemMessage):
        candidate = message.data.get("session_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def message_to_event(message: Any) -> dict[str, Any]:
    """Translate one SDK message into the server's event dict."""
    if isinstance(message, AssistantMessage):
        payload: dict[str, Any] = {
            "content": [block_to_dict(block) for block in message.content],
            "role": "assistant",
            "model": message.model,
        }
        if message.message_id is not None:
            payload["id"] = message.message_id
        if message.usage is not None:
            payload["usage"] = message.usage
        if message.stop_reason is not None:
            payload["stop_reason"] = message.stop_reason
        event: dict[str, Any] = {"type": "assistant", "message": payload}
        if message.parent_tool_use_id is not None:
            event["parent_tool_use_id"] = message.parent_tool_use_id
        if message.error is not None:
            event["error"] = _plain(message.error)
        return _with_session(event, message)

    if isinstance(message, UserMessage):
        event = {
            "type": "user",
            "message": {"role": "user", "content": _content_to_list(message.content)},
        }
        if message.parent_tool_use_id is not None:
            event["parent_tool_use_id"] = message.parent_tool_use_id
        if message.tool_use_result is not None:
            event["tool_use_result"] = message.tool_use_result
        return _with_session(event, message)

    if isinstance(message, ResultMessage):
        event = {
            "type": "result",
            "subtype": message.subtype,
            "duration_ms": message.duration_ms,
            "duration_api_ms": message.duration_api_ms,
            "is_error": message.is_error,
            "num_turns": message.num_turns,
        }
        for name in ("result", "stop_reason", "total_cost_usd", "usage"):
            value = getattr(message, name, None)
            if value is not None:
                event[name] = value
        return _with_session(event, message)

    if isinstance(message, SystemMessage):
        # ``data`` is the raw CLI payload, so spreading it keeps fields the app
        # may already read (``tools``, ``cwd``, ``model`` on init, …).
        event = {"type": "system", "subtype": message.subtype}
        event.update(message.data)
        event["type"] = "system"
        return _with_session(event, message)

    return _with_session(_plain(message), message)


def _with_session(event: dict[str, Any], message: Any) -> dict[str, Any]:
    if "session_id" not in event:
        session_id = session_id_of(message)
        if session_id is not None:
            event["session_id"] = session_id
    return event


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        data = asdict(value)
        data.setdefault("type", _snake(type(value).__name__))
        return data
    return {"type": _snake(type(value).__name__), "value": str(value)}


def _snake(name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            out.append("_")
        out.append(char.lower())
    return "".join(out)
