"""Streaming service for websocket chat turns."""

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

from .chat_event_utils import extract_assistant_text, format_tool_result_content
from approvals.approval_service import (
    default_approval_expires_at,
    request_approval_for_operation,
)
from agent.agent_store import get_agent_store
from llm.llm_session import LlmSession
from llm.claude_usage import fetch_claude_usage_snapshot, merge_usage_for_display
from core.config import get_config
from core.database import get_usage_db
from projects.project_manager import get_project_manager

logger = logging.getLogger(__name__)
APP_EVENT_SCHEMA_VERSION = 1
PROVIDER_EVENT_SCHEMA_VERSION = 1


def _build_flutter_test_mcp_context(project_name: str) -> str | None:
    """Build Flutter runtime MCP context reminder if a Flutter app is running.

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

A Flutter app is currently running and you can interact with it using Flutter Test MCP.

**Device:** {device_id}
**VM Service URI:** {vm_uri}

### Available Commands:

1. **Connect to app** (required before other commands):
   ```
   mcp__flutter_test__connect(uri: "{vm_uri}")
   ```

2. **Hot Reload** (apply code changes):
   ```
   mcp__flutter_test__hot_reload()
   ```

3. **Get interactive elements** (find tappable widgets):
   ```
   mcp__flutter_test__get_interactive_elements()
   ```

4. **Tap element** (by key, text, or coordinates):
   ```
   mcp__flutter_test__tap(text: "Button Text")
   mcp__flutter_test__tap(key: "my_button_key")
   ```

5. **Enter text** (requires widget key):
   ```
   mcp__flutter_test__enter_text(key: "email_field", input: "test@example.com")
   ```

6. **Take screenshot**:
   ```
   mcp__flutter_test__take_screenshots()
   ```

**Workflow:** connect -> get_interactive_elements -> tap/enter_text -> take_screenshots
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


# Tools that only observe the *local* workspace. Nothing here mutates the
# workspace, the host, or the repository; the worst any of them can do is
# *look at* something it should not, and that is exactly what the path guard
# and the secret classifier exist to catch (see `_approval_target_details`
# below, which promotes the tool's own target into the policy details so
# those classifiers have something to inspect).
READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read",
        "glob",
        "grep",
        "ls",
        "notebookread",
        "todoread",
        "todowrite",
    }
)

# WebFetch/WebSearch read too, but what they read is a URL, and that changes
# who can choose it. The safety valve under `file.read` is the path guard,
# and a URL never reaches it: `decide_policy("file.read", details={})` has
# nothing to inspect and returns `allow`. Filing these two as read-only would
# therefore auto-approve every outbound request an unattended run makes,
# including a URL the model was talked into by content it fetched earlier —
# the target of a prompt injection is chosen by the attacker, not the user.
# `network.external` is `CONFIRM_EACH`, which is the honest default for
# "leaves this machine".
NETWORK_TOOL_NAMES: frozenset[str] = frozenset({"webfetch", "websearch"})


def _approval_operation_for_tool(tool_name: Any) -> str:
    """Map provider-native tool names to Code Bridge policy operations.

    The literal strings this returns are the entire set of operations a
    standing policy rule can ever match for an LLM tool-permission prompt —
    see ``LLM_TOOL_APPROVAL_OPERATIONS`` below, which must list exactly these
    values. A rule created for anything outside that set (the dashboard
    used to also offer ``browser.control``) looks like a working standing
    permission and is not: it can never be consulted, because no call site
    ever asks for approval under that operation name.

    The read-only branch exists because its absence inverted the whole
    product. Read/Glob/Grep used to fall through to ``provider.tool``, which
    is in no policy set at all, so it hit ``decide_policy``'s unknown branch
    and came back ``confirm_each`` — "Unknown operations require explicit
    confirmation". Meanwhile ``process.terminal`` *is* a named operation, so
    a single standing allow rule (the natural thing to create the first time
    a run stalls on a shell command) silently auto-approved every subsequent
    ``Bash``. The result asked permission to read a file and did not ask to
    run a shell command. Mapping the read-only tools to ``file.read`` — an
    ``ALLOW_OPERATIONS`` member and already a ``_PATH_OPERATIONS`` member, so
    the path guard still runs over the target — puts the low-risk half back
    where it belongs without weakening anything. ``Bash`` deliberately stays
    on ``process.terminal``: the inversion was the Read side, and a user's
    existing standing rule stays theirs.
    """
    normalized = str(tool_name or "").strip().lower()
    if normalized in {"bash", "shell", "terminal", "run_command"}:
        return "process.terminal"
    if normalized in {"edit", "multiedit", "write", "notebookedit"}:
        return "file.write"
    if normalized in {"git", "git_commit", "git_push"}:
        return "git.commit"
    if normalized in NETWORK_TOOL_NAMES:
        return "network.external"
    if normalized in READ_ONLY_TOOL_NAMES:
        return "file.read"
    return "provider.tool"


# The exhaustive set of operations `_approval_operation_for_tool` can produce.
# The dashboard's "Unattended permissions" form used to hand-copy a second,
# independent list of operations into an HTML <select> instead of reading
# this one, and the two drifted: the form offered `file.read` and
# `browser.control` (which this function did not then return, so a rule for
# them never matched anything) and omitted `git.commit` and `provider.tool`
# (which it did return, so the common case of a scheduled run stalling on a
# plain tool read had no way to be pre-authorized at all). Anything that
# builds that option list must derive it from here rather than copying it
# again, or the same drift happens the next time a branch is added or
# renamed. `tests/test_llm_tool_approval_operations.py` enforces this by
# reading `_approval_operation_for_tool`'s source and diffing its literal
# return values against this tuple.
#
# `file.read` is on this list now because the read-only tools map to it. Note
# what that does *not* retroactively make true of the old dashboard form: a
# rule created back then still could not match anything, because at the time
# nothing asked for approval under that name. The fix was to make the runtime
# ask, not to re-offer the option.
LLM_TOOL_APPROVAL_OPERATIONS: tuple[str, ...] = (
    "process.terminal",
    "file.write",
    "file.read",
    "network.external",
    "git.commit",
    "provider.tool",
)


# Which input key names the filesystem target of each tool, per tool. The
# distinction that matters here is `pattern`: for `Glob` it is a path glob
# (`src/**/*.ts`, and just as easily `~/.ssh/*`), so it must reach the path
# guard; for `Grep` it is a *regular expression* over file contents, and
# feeding a regex to the path guard would produce escalations that mean
# nothing. Guessing one rule for both keys gets one of them wrong, so this
# table is per-tool rather than a single key list.
_TOOL_PATH_INPUT_KEYS: dict[str, tuple[str, ...]] = {
    "read": ("file_path",),
    "write": ("file_path",),
    "edit": ("file_path",),
    "multiedit": ("file_path",),
    "notebookread": ("notebook_path",),
    "notebookedit": ("notebook_path",),
    "glob": ("path", "pattern"),
    "grep": ("path",),
    "ls": ("path",),
}

# For a tool this module has never heard of, only keys whose name states they
# are a path are trusted. An unknown tool's `pattern`/`query`/`url` could be
# anything.
_GENERIC_PATH_INPUT_KEYS: tuple[str, ...] = ("file_path", "path", "notebook_path")


def _approval_target_details(
    tool_name: Any,
    tool_input: Any,
    *,
    workspace_root: str | None,
) -> dict[str, Any]:
    """Lift a tool's own target into the keys the policy classifiers read.

    This is the safety valve that makes mapping the read-only tools to
    ``file.read`` (an ``ALLOW_OPERATIONS`` member) acceptable. ``decide_policy``
    only runs :mod:`policy.path_guard` over ``details["path"]`` /
    ``details["paths"]``; the tool's target used to live one level down under
    ``details["input"]["file_path"]``, where no classifier looks. Without this
    promotion the allow would be an unconditional allow, and reading
    ``~/.ssh/id_rsa`` would be auto-approved. With it, ``_apply_classifiers``
    escalates that same request to ``desktop_only`` — escalation applies to a
    base effect of ``allow`` exactly as it does to any other.

    Relative targets are resolved against ``workspace_root`` rather than the
    server process's own CWD, which has nothing to do with the project the
    model is working in.
    """
    if not isinstance(tool_input, dict):
        return {}

    normalized = str(tool_name or "").strip().lower()
    keys = _TOOL_PATH_INPUT_KEYS.get(normalized, _GENERIC_PATH_INPUT_KEYS)

    paths: list[str] = []
    for key in keys:
        value = tool_input.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = value.strip()
        if (
            workspace_root
            and not candidate.startswith("~")
            and not os.path.isabs(candidate)
        ):
            candidate = os.path.join(workspace_root, candidate)
        if candidate not in paths:
            paths.append(candidate)

    if not paths:
        return {}
    return {"path": paths[0], "paths": paths}


# Normalized approval-card vocabulary. The server knows every tool's input
# schema; the app does not, and should not have to learn nine of them to say
# "the agent wants to read a file". `action` is a stable key the app maps to
# its own localized sentence — deliberately not a sentence itself, because the
# app ships four locales (en/ko/ja/zh) and a server-side Korean string would
# be wrong in three of them.
_TOOL_DISPLAY_ACTIONS: dict[str, str] = {
    "read": "read_file",
    "notebookread": "read_file",
    "bash": "run_command",
    "shell": "run_command",
    "terminal": "run_command",
    "run_command": "run_command",
    "edit": "write_file",
    "multiedit": "write_file",
    "write": "write_file",
    "notebookedit": "write_file",
}

# The key that names each normalized action's target, in priority order.
_DISPLAY_TARGET_KEYS: dict[str, tuple[str, ...]] = {
    "read_file": ("file_path", "notebook_path", "path"),
    "run_command": ("command",),
    "write_file": ("file_path", "notebook_path", "path"),
}

# Fallback target probe for a tool with no normalized action. Same key order
# as `agent/task_orchestrator.py::_approval_tool_target`, which already picks
# the parked step's `tool_target` this way — one vocabulary, not two.
_FALLBACK_TARGET_KEYS: tuple[str, ...] = (
    "file_path",
    "path",
    "command",
    "url",
    "pattern",
    "notebook_path",
    "query",
)


def _run_workspace_root(run_id: str) -> str | None:
    """The registered cwd of an agent run, or ``None`` if it has none.

    Never raises: a store lookup failing here must not turn into a failed
    permission prompt. Returning ``None`` costs an extra confirmation, which
    is the right way to be wrong.
    """
    try:
        run = get_agent_store().get_run(run_id)
    except Exception:  # noqa: BLE001 - policy must not fail on a store read
        logger.warning("[chat_stream] workspace root lookup failed run=%s", run_id)
        return None
    if not isinstance(run, dict):
        return None
    cwd = run.get("cwd")
    return cwd.strip() if isinstance(cwd, str) and cwd.strip() else None


def _run_agent_identity(run_id: str) -> dict[str, str] | None:
    """The real agent behind a run, as ``{"id", "name"}``, or ``None``.

    This is what lets a standing rule outlive the run it was granted in: a
    scheduled fire is a new run, so an ``agent:`` scope is the narrowest thing
    that is still there next time
    (``approvals/approval_service.py::standing_rule_scope_for_request``).

    ``None`` is returned for anything that is not one nameable agent, and that
    is the whole safety of the feature:

    * **Pseudo-agents.** Interactive chat files its runs under
      ``agent_adhoc_dev`` (``routes/chat_ws.py:285``) and migrated rows under
      ``agent_legacy_chat`` (``core/database.py:605``). Both are shared by
      every chat on the server, exactly like the ``__global__`` project
      sentinel, so a rule there would be a near-global grant wearing an
      agent's name. Chat therefore keeps today's behaviour untouched.
    * **Anything unresolvable.** A missing run, a run with no agent, a store
      that raises — all fall through to ``None``, which drops the scope back
      to ``run:``. Failing that way costs a repeated prompt; failing the other
      way would grant a permission to the wrong agent.

    The run row carries ``agent_id`` directly (``agent/agent_store.py:520``,
    set from the task's ``assigned_agent_id`` at
    ``agent/task_orchestrator.py:421``); the task is consulted only as a
    fallback for a run that predates that column being written.
    """
    try:
        store = get_agent_store()
        run = store.get_run(run_id)
        if not isinstance(run, dict):
            return None
        agent_id = run.get("agent_id")
        if not (isinstance(agent_id, str) and agent_id.strip()):
            task_id = run.get("task_id")
            task = store.get_task(task_id) if isinstance(task_id, str) and task_id else None
            agent_id = task.get("assigned_agent_id") if isinstance(task, dict) else None
        if not (isinstance(agent_id, str) and agent_id.strip()):
            return None
        agent = store.get_agent(agent_id.strip())
        if not isinstance(agent, dict) or agent.get("is_pseudo"):
            return None
        identity = {"id": agent_id.strip()}
        name = agent.get("name")
        if isinstance(name, str) and name.strip():
            identity["name"] = name.strip()
        return identity
    except Exception:  # noqa: BLE001 - policy must not fail on a store read
        logger.warning("[chat_stream] agent identity lookup failed run=%s", run_id)
        return None


def _approval_display(tool_name: Any, tool_input: Any) -> dict[str, Any]:
    """``{"action", "target"}`` for the approval card.

    Unrecognised tools keep their own name as the action, verbatim: an MCP
    tool called ``linear__create_issue`` is more informative shown as itself
    than flattened into a generic bucket, and the app already has a JSON
    fallback for an action it cannot localize. ``target`` is ``None`` rather
    than a guess when nothing in the input names one.
    """
    raw_name = str(tool_name).strip() if isinstance(tool_name, str) else ""
    normalized = raw_name.lower()
    action = _TOOL_DISPLAY_ACTIONS.get(normalized) or raw_name or "unknown"

    target: str | None = None
    if isinstance(tool_input, dict):
        keys = _DISPLAY_TARGET_KEYS.get(action, _FALLBACK_TARGET_KEYS)
        for key in keys:
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                target = value.strip()
                break

    return {"action": action, "target": target}


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
        display = _approval_display(tool_name, tool_input)
        agent_run_id = getattr(websocket, "agent_run_id", None)
        if isinstance(agent_run_id, str) and agent_run_id.strip():
            # The run's registered cwd is the workspace root the path guard
            # measures against. Without it every path reads as
            # `workspace_external` (confirm_each) and mapping the read-only
            # tools to `file.read` would change nothing — the whole point is
            # that reading *inside the project you pointed the agent at* is
            # the low-risk case, and reading outside it is not. When the run
            # has no cwd we pass none and the guard's own default (treat as
            # external, confirm) applies: fail closed, never open.
            workspace_root = _run_workspace_root(agent_run_id)
            details: dict[str, Any] = {
                "project_name": project_name,
                "provider_id": state.provider_id,
                "session_id": state.session_id,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "input": tool_input if isinstance(tool_input, dict) else {},
                "provider_request_id": request_id,
                "display": display,
            }
            if workspace_root:
                details["workspace_root"] = workspace_root
            # Which agent asked. Only present for a real (non-pseudo) agent,
            # and it is what a standing rule can be anchored to so the grant
            # survives to the agent's next scheduled fire. Absent for chat,
            # which keeps resolving exactly as it did.
            agent_identity = _run_agent_identity(agent_run_id)
            if agent_identity:
                details["agent_id"] = agent_identity["id"]
                if agent_identity.get("name"):
                    # Display only — the card names the agent the rule would
                    # cover instead of printing an opaque `agent_…` id.
                    details["agent_name"] = agent_identity["name"]
            details.update(
                _approval_target_details(
                    tool_name,
                    tool_input,
                    workspace_root=workspace_root,
                )
            )
            approval_result = request_approval_for_operation(
                operation=_approval_operation_for_tool(tool_name),
                run_id=agent_run_id,
                actor={"type": "agent_session"},
                details=details,
                # A tool call parks the whole run until someone answers it, so
                # the question has to stop being askable eventually — see
                # `approvals.approval_service.default_approval_expires_at` and
                # the scheduler's expiry sweep, which resumes the parked turn
                # down the deny path once this deadline passes.
                expires_at=default_approval_expires_at(),
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
                "display": display,
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
    logger.info(
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
        flutter_test_mcp_ctx = _build_flutter_test_mcp_context(project_name)
        if flutter_test_mcp_ctx:
            user_message = f"{user_message}\n\n{flutter_test_mcp_ctx}"

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
