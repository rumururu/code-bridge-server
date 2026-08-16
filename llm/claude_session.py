"""Claude Code session management via the official claude-agent-sdk.

This used to drive the CLI directly: a local WebSocket server plus
``claude --sdk-url ws://127.0.0.1:…`` so the CLI dialled back in. Claude Code
now refuses that flag for non-Anthropic hosts ("--sdk-url rejected: host
127.0.0.1 is not an approved Anthropic endpoint"), so every turn died on a
15-second connect timeout. The SDK is the supported wrapper around the same
CLI — it owns the transport and absorbs protocol changes, and it authenticates
exactly as the CLI does, so this keeps running on the user's Claude
subscription rather than metered API billing.
"""

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from claude_agent_sdk.types import AgentDefinition

from .claude_sdk_events import message_to_event, session_id_of
from .llm_session import LlmSession, CliAgentDefinition

logger = logging.getLogger(__name__)


ASK_USER_SYSTEM_PROMPT = (
    "When you need a yes/no or multiple-choice response from the human user, "
    "DO NOT write the question as free-form prose. Instead, emit EXACTLY this "
    'tag on its own line: <ask_user question="YOUR_QUESTION_HERE" '
    'options="A|B|C"/>. Rules: (1) use double quotes; (2) separate options '
    "with the pipe character; (3) the tag must be on its own line with no "
    "leading or trailing prose on the same line; (4) keep the question in the "
    "same language the user is conversing in; (5) do not also restate the "
    "options as a numbered list — the tag is the only UI. Example: "
    '<ask_user question="Proceed with the migration?" options="Yes|No"/>. '
    "For purely informational messages that do not require a choice, write "
    "prose as usual and do NOT emit the tag."
)


#: The reasoning-effort values a Claude agent definition accepts. A definition
#: file is free to write something else; an unrecognized value is dropped
#: rather than sent, because the SDK would reject the whole definition and the
#: run would fail over a decorative hint.
_AGENT_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})


@dataclass
class ClaudeSession(LlmSession):
    """Manage one long-lived Claude CLI process with real-time control responses."""

    project_path: str
    default_permission_mode: str = "default"
    model: str | None = None
    #: When set, this session runs *as* the named Claude Code agent this
    #: definition describes, instead of as a plain Claude Code session. This is
    #: the only place a definition can be run: it is the mechanism that makes
    #: the declared tools real, and no other provider has one — the factory
    #: refuses them rather than dropping it. See :meth:`_build_options`.
    cli_agent: CliAgentDefinition | None = None
    #: MCP servers this session should be able to reach, keyed by server name,
    #: in the shape ``ClaudeAgentOptions.mcp_servers`` takes
    #: (``{"type": "stdio", "command": …}`` / ``{"type": "http", "url": …}``).
    #: Set by the orchestrator from the agent's declared tools; see
    #: :meth:`set_mcp_servers`. ``None`` means "say nothing", which leaves the
    #: CLI's own configuration in force — not "no servers".
    mcp_servers: dict[str, Any] | None = None
    _claude_path: str = field(default="", init=False)
    _session_id: str | None = field(default=None, init=False)
    _client: ClaudeSDKClient | None = field(default=None, init=False)
    _pump_task: asyncio.Task[None] | None = field(default=None, init=False)
    _event_queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue, init=False)
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _turn_in_progress: bool = field(default=False, init=False)
    _pending_permission_request: dict[str, Any] | None = field(default=None, init=False)
    _pending_permission_future: asyncio.Future[Any] | None = field(default=None, init=False)

    # If the stored project path does not exist on disk we start the session
    # in a safe fallback cwd and stash a short note so the LLM receives the
    # context on its first user turn. Without this the CLI subprocess would
    # die with ENOENT before any user message could be exchanged.
    _fallback_note: str | None = field(default=None, init=False)
    _fallback_note_delivered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Locate Claude executable."""
        self._claude_path = shutil.which("claude") or ""
        if not self._claude_path:
            for path in (
                os.path.expanduser("~/.local/bin/claude"),
                "/usr/local/bin/claude",
                "/opt/homebrew/bin/claude",
            ):
                if os.path.exists(path):
                    self._claude_path = path
                    break

    @property
    def provider_id(self) -> str:
        """Return the provider identifier."""
        return "anthropic"

    @property
    def is_running(self) -> bool:
        """Whether the SDK client is connected."""
        return self._client is not None

    @property
    def session_id(self) -> str | None:
        """Claude conversation session id."""
        return self._session_id

    async def resume_session(self, session_id: str) -> None:
        """Pin a past Claude session id so the next turn resumes that context.

        Closes any running CLI process so the new `--resume <id>` flag takes
        effect on the next send_message.
        """
        next_id = session_id.strip() if isinstance(session_id, str) else ""
        if not next_id:
            return
        if self._session_id == next_id and self.is_running:
            return
        self._session_id = next_id
        if self.is_running:
            await self.close()

    @property
    def has_pending_permission_denials(self) -> bool:
        """Backward-compatible flag for pending permission request."""
        return self._pending_permission_request is not None

    @property
    def has_pending_permission(self) -> bool:
        """Whether a live turn is parked on an unanswered permission prompt.

        Both halves matter. ``_pending_permission_request`` alone is the
        request; ``_turn_in_progress`` is what
        ``_respond_to_pending_permission`` checks before it hands a decision
        back to the parked ``can_use_tool`` callback, and it refuses with "No
        Claude turn in progress" otherwise. The orchestrator asks this before
        resuming a run from an approval, so answering yes when the turn is gone
        would turn a resume into an error event instead of a re-execution.
        """
        return self._turn_in_progress and self._pending_permission_request is not None

    def _resolve_start_cwd(self) -> tuple[str, str | None]:
        """Pick a safe working directory for the Claude subprocess.

        Returns ``(cwd, fallback_note)``. ``fallback_note`` is a Korean system
        message the caller should prepend to the first user message whenever
        the stored path was unreachable, so the LLM can help the user fix the
        project metadata via the Dashboard API instead of just failing.
        """
        if self.project_path and os.path.isdir(self.project_path):
            return self.project_path, None

        fallback = os.path.expanduser("~")
        note = (
            "[System notice] This Claude session started in the Mac home "
            f"directory `{fallback}` because the project path "
            f"`{self.project_path}` does not exist on this Mac.\n\n"
            "Reason: The path stored in the Code Bridge server DB "
            "(projects.path) does not match any real filesystem path on this "
            "Mac. This happens when the DB was copied from another Mac or "
            "when the project was moved.\n\n"
            "Steps to resolve:\n"
            "1. Use bash `ls ~/VSCodeProject ~/AndroidStudioProjects "
            "~/Projects ~/code ~/Documents` etc. to locate the project folder.\n"
            "2. Run `curl -s http://localhost:8766/api/projects` to inspect "
            "the currently registered project list.\n"
            "3. Send `{\"path\": \"/real/path\"}` JSON via "
            "`curl -X PUT http://localhost:8766/api/projects/<PROJECT_NAME>` "
            "to update the DB.\n"
            "4. After the update, tell the user to tap \"Retry\".\n\n"
            "The dashboard port (8766) is localhost-only and does not require "
            "an API key. Before changing any value, show your proposal to "
            "the user and get approval first."
        )
        return fallback, note

    def _build_options(self, permission_mode: str | None = None) -> ClaudeAgentOptions:
        """Build SDK options for one connection.

        Replaces the hand-assembled CLI argv. The old command passed
        ``--sdk-url`` so the CLI would dial back into a WebSocket server this
        class ran; Claude Code now rejects that flag for non-Anthropic hosts,
        which killed every Claude turn. The SDK owns the transport instead.

        When :attr:`cli_agent` is set, the turn runs as that named agent: the
        definition goes in ``options.agents`` (the SDK sends it in the
        initialize request) and ``--agent <name>`` selects it. Both halves are
        needed and neither is decorative — the definition is what makes the
        declared ``tools`` real (a session given ``tools: Read, Glob`` answers
        that it has no Bash), and it is also the only way a *plugin* agent
        runs at all, since ``--agent`` on its own resolves user and project
        agents only. Passing the prompt as ordinary text instead, which is what
        this used to do, silently gives a read-only agent write access.
        """
        cwd, fallback_note = self._resolve_start_cwd()
        if fallback_note is not None:
            self._fallback_note = fallback_note
            self._fallback_note_delivered = False

        resolved_mode = (permission_mode or self.default_permission_mode).strip()
        options = ClaudeAgentOptions(
            cwd=cwd,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": ASK_USER_SYSTEM_PROMPT,
            },
            can_use_tool=self._on_can_use_tool,
        )
        if resolved_mode:
            options.permission_mode = resolved_mode
        if isinstance(self.model, str) and self.model.strip():
            options.model = self.model.strip()
        if self._session_id:
            options.resume = self._session_id
        self._apply_mcp_servers(options)
        self._apply_cli_agent(options)
        return options

    def _apply_mcp_servers(self, options: ClaudeAgentOptions) -> None:
        """Hand the declared MCP servers to the SDK, if there are any.

        ``ClaudeAgentOptions.mcp_servers`` exists in claude-agent-sdk 0.2.128
        (``types.py``: ``dict[str, McpServerConfig] | str | Path``) and is the
        only route from an agent's declared tools to the process that would
        run them — without it the agent's ``tools_json`` never left the
        database and the run used whatever the user's CLI happened to have.

        ``allowed_tools`` is deliberately *not* set alongside it. In this SDK
        that field pre-approves tool calls, so listing ``mcp__x__y`` there
        would let the injected servers run without the approval round-trip
        that :meth:`_on_can_use_tool` exists to perform. The servers are made
        reachable; each call still asks.
        """
        servers = self.mcp_servers
        if not servers:
            return
        options.mcp_servers = dict(servers)

    async def set_mcp_servers(self, servers: dict[str, Any] | None) -> None:
        """Set the MCP servers for subsequent turns.

        Options are built once per connection, so a live client is closed when
        the set changes — the next turn reconnects with the new servers.
        Mutating the field alone would leave a running session claiming
        servers it was never started with.
        """
        next_servers = dict(servers) if servers else None
        if next_servers == self.mcp_servers:
            return
        self.mcp_servers = next_servers
        if self.is_running:
            await self.close()

    def _apply_cli_agent(self, options: ClaudeAgentOptions) -> None:
        """Attach the agent definition and select it, if there is one."""
        cli_agent = self.cli_agent
        if cli_agent is None:
            return
        definition = AgentDefinition(
            description=cli_agent.description or cli_agent.name,
            prompt=cli_agent.prompt,
        )
        if cli_agent.tools:
            definition.tools = list(cli_agent.tools)
        # `model: inherit` is meaningful here in a way it is not in the agent
        # record: at this layer there really is a parent session to inherit
        # from, so it is passed through rather than resolved to an id.
        if cli_agent.model:
            definition.model = cli_agent.model
        if cli_agent.effort in _AGENT_EFFORT_LEVELS:
            definition.effort = cli_agent.effort
        options.agents = {cli_agent.name: definition}
        options.extra_args = {**(options.extra_args or {}), "agent": cli_agent.name}

    async def _ensure_client(self, permission_mode: str | None = None) -> None:
        """Connect the SDK client if it is not already live."""
        async with self._start_lock:
            if self.is_running:
                return

            self._turn_in_progress = False
            self._pending_permission_request = None
            self._pending_permission_future = None
            self._event_queue = asyncio.Queue()

            options = self._build_options(permission_mode)
            logger.warning(
                "[claude_session] connect project_path=%r cwd=%r resume=%s",
                self.project_path,
                options.cwd,
                "yes" if options.resume else "no",
            )

            client = ClaudeSDKClient(options=options)
            try:
                await client.connect()
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as an error event
                raise RuntimeError(f"Claude SDK connect failed: {exc}") from exc

            self._client = client
            self._pump_task = asyncio.create_task(self._pump_messages())

    async def _pump_messages(self) -> None:
        """Feed converted SDK messages into the turn queue.

        One long-lived reader per connection: ``send_message`` and the
        permission responders all consume ``_event_queue``, so the permission
        callback (which the SDK invokes on its own task) and the message
        stream stay in one ordered channel.
        """
        client = self._client
        if client is None:
            return
        try:
            async for message in client.receive_messages():
                session_id = session_id_of(message)
                if session_id:
                    self._session_id = session_id
                await self._event_queue.put(message_to_event(message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the active turn
            logger.warning("[claude_session] message pump ended: %s", exc)
            await self._event_queue.put(
                {"type": "session_closed", "returncode": None, "stderr": str(exc)}
            )

    async def _on_can_use_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Bridge the SDK's permission callback to the app's approval round-trip.

        The SDK expects a returned decision; the app expects a
        ``control_request`` event and answers later over its own websocket.
        Park the callback on a future, publish the request, and let
        ``approve_pending_permissions_and_retry`` / ``deny_pending_permissions``
        resolve it.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PermissionResultAllow | PermissionResultDeny] = loop.create_future()
        request_id = uuid.uuid4().hex
        event = {
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "can_use_tool",
                "tool_name": tool_name,
                "input": tool_input,
                "tool_use_id": context.tool_use_id,
            },
        }
        self._pending_permission_future = future
        await self._event_queue.put(event)
        return await future

    def _settle_pending_permission(
        self,
        decision: PermissionResultAllow | PermissionResultDeny,
    ) -> bool:
        """Hand a decision back to the parked ``can_use_tool`` callback."""
        future = self._pending_permission_future
        if future is None or future.done():
            return False
        future.set_result(decision)
        self._pending_permission_future = None
        return True

    @staticmethod
    def _enrich_with_call_id(event: dict[str, Any]) -> dict[str, Any]:
        """Add top-level ``call_id`` for tool_use / tool_result events.

        The Anthropic Messages API places the tool call identifier on the
        first content block of an ``assistant`` event:

        - ``message.content[0].id`` for ``tool_use`` blocks
        - ``message.content[0].tool_use_id`` for ``tool_result`` blocks

        Surface that identifier as a top-level ``call_id`` so the generic
        ``AgentTaskRunSink`` can persist it into ``agent_events.call_id``
        without provider-specific knowledge.

        Always returns a new dict when modifying so the SDK-owned event
        object stays immutable for any other consumer.
        """
        if not isinstance(event, dict):
            return event
        if event.get("type") != "assistant":
            return event
        message = event.get("message")
        if not isinstance(message, dict):
            return event
        content = message.get("content")
        if not isinstance(content, list) or not content:
            return event
        first_block = content[0]
        if not isinstance(first_block, dict):
            return event
        block_type = first_block.get("type")
        if block_type == "tool_use":
            block_id = first_block.get("id")
            if isinstance(block_id, str) and block_id:
                return {**event, "call_id": block_id}
        elif block_type == "tool_result":
            tool_use_id = first_block.get("tool_use_id")
            if isinstance(tool_use_id, str) and tool_use_id:
                return {**event, "call_id": tool_use_id}
        return event

    async def _stream_until_pause_or_result(self) -> AsyncGenerator[dict[str, Any], None]:
        """Yield events until permission pause or result event."""
        while True:
            event = await self._event_queue.get()
            event_type = event.get("type")

            if event_type == "control_request":
                request = event.get("request", {})
                if isinstance(request, dict) and request.get("subtype") == "can_use_tool":
                    self._pending_permission_request = event
                    yield event
                    break

            if event_type == "result":
                self._turn_in_progress = False
                self._pending_permission_request = None
                yield event
                break

            if event_type == "session_closed":
                self._turn_in_progress = False
                self._pending_permission_request = None
                yield {
                    "type": "error",
                    "error": {
                        "message": (
                            f"Claude session ended (code {event.get('returncode')})"
                            + (
                                f": {event.get('stderr')}"
                                if isinstance(event.get("stderr"), str) and event.get("stderr")
                                else ""
                            )
                        )
                    },
                }
                break

            yield self._enrich_with_call_id(event)

    async def send_message(
        self,
        message: str,
        permission_mode: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Start a new Claude turn.

        If the message contains [Uploaded Attachments] with image paths,
        they will be converted to multimodal content blocks.
        """
        if not message.strip():
            yield {"type": "error", "error": {"message": "Message content is empty"}}
            return

        try:
            if permission_mode and permission_mode.strip() and permission_mode != self.default_permission_mode:
                await self.close()
                self.default_permission_mode = permission_mode

            await self._ensure_client(permission_mode)

            if self._turn_in_progress:
                yield {"type": "error", "error": {"message": "Another Claude turn is already in progress"}}
                return

            # Inject the fallback-cwd note on the very first turn so the LLM
            # has context about the missing project path and can drive a fix.
            if self._fallback_note and not self._fallback_note_delivered:
                message = f"{self._fallback_note}\n\n---\n\n{message}"
                self._fallback_note_delivered = True

            # Build content - may be string or multimodal list
            content = self._build_message_content(message)

            async with self._turn_lock:
                self._turn_in_progress = True
                self._pending_permission_request = None

                client = self._client
                if client is None:
                    yield {"type": "error", "error": {"message": "Claude session is not connected"}}
                    return
                await client.query(content)

                async for event in self._stream_until_pause_or_result():
                    yield event
        except (OSError, ConnectionError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
            self._turn_in_progress = False
            yield {"type": "error", "error": {"message": str(exc)}}

    def _build_message_content(self, message: str) -> str | list[dict[str, Any]]:
        """Parse attachments and build multimodal content if images are present."""
        try:
            from attachment_parser import build_multimodal_content
            return build_multimodal_content(message, self.project_path)
        except (ImportError, ValueError, TypeError, AttributeError, OSError):
            # Fallback to plain text if parsing fails
            return message

    async def _respond_to_pending_permission(
        self,
        allow: bool,
        deny_message: str = "Permission denied by user.",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Respond to pending can_use_tool and continue current turn."""
        if not self._turn_in_progress:
            yield {"type": "error", "error": {"message": "No Claude turn in progress"}}
            return

        pending = self._pending_permission_request
        if pending is None:
            yield {"type": "error", "error": {"message": "No pending permission request"}}
            return

        request = pending.get("request", {})
        if not isinstance(request, dict):
            yield {"type": "error", "error": {"message": "Invalid pending permission request state"}}
            return

        tool_input = request.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}

        decision: PermissionResultAllow | PermissionResultDeny
        if allow:
            decision = PermissionResultAllow(updated_input=tool_input)
        else:
            decision = PermissionResultDeny(message=deny_message, interrupt=False)

        if not self._settle_pending_permission(decision):
            self._turn_in_progress = False
            self._pending_permission_request = None
            yield {
                "type": "error",
                "error": {"message": "Permission request is no longer awaiting a decision"},
            }
            return

        self._pending_permission_request = None
        async for event in self._stream_until_pause_or_result():
            yield event

    async def approve_pending_permissions_and_retry(self) -> AsyncGenerator[dict[str, Any], None]:
        """Approve permission prompt and continue current turn."""
        async for event in self._respond_to_pending_permission(allow=True):
            yield event

    async def deny_pending_permissions(
        self,
        message: str = "Permission denied by user.",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Deny permission prompt and continue current turn."""
        async for event in self._respond_to_pending_permission(allow=False, deny_message=message):
            yield event

    async def close(self) -> None:
        """Disconnect the SDK client and stop the message pump."""
        pump = self._pump_task
        if pump is not None and not pump.done():
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        self._pump_task = None

        # Release a parked permission callback so the SDK's own task cannot
        # outlive the connection waiting on a decision that will never come.
        self._settle_pending_permission(
            PermissionResultDeny(message="Session closed.", interrupt=True)
        )

        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as exc:  # noqa: BLE001 - teardown is best effort
                logger.debug("Error disconnecting Claude SDK client: %s", exc)

        self._turn_in_progress = False
        self._pending_permission_request = None
        self._pending_permission_future = None

    async def abort_current_turn(self) -> bool:
        """Interrupt the running turn.

        Returns True if an interrupt was delivered, False if nothing was running.
        """
        if not self._turn_in_progress:
            return False

        client = self._client
        if client is None:
            self._turn_in_progress = False
            return False

        # A turn parked on a permission prompt has no model work to interrupt —
        # deny it instead so the SDK callback unblocks and the turn unwinds.
        if self._settle_pending_permission(
            PermissionResultDeny(message="Turn aborted by user.", interrupt=True)
        ):
            self._turn_in_progress = False
            self._pending_permission_request = None
            return True

        try:
            await client.interrupt()
        except Exception as exc:  # noqa: BLE001 - reported via the return value
            logger.debug("Claude interrupt failed: %s", exc)
            self._turn_in_progress = False
            return False

        self._turn_in_progress = False
        self._pending_permission_request = None
        return True

    async def set_model(self, model: str | None) -> None:
        """Set default model for subsequent turns.

        If the underlying Claude process is already running, restart it so the
        CLI model flag is applied consistently.
        """
        next_model = model.strip() if isinstance(model, str) and model.strip() else None
        current_model = self.model.strip() if isinstance(self.model, str) and self.model.strip() else None
        if next_model == current_model:
            return

        self.model = next_model
        client = self._client
        if client is None:
            return
        try:
            # The SDK changes the model on the live session; the old transport
            # had to restart the CLI to re-pass --model.
            await client.set_model(next_model)
        except Exception as exc:  # noqa: BLE001 - fall back to a reconnect
            logger.debug("Live model switch failed, reconnecting: %s", exc)
            await self.close()


class SessionManager:
    """Manage LLM sessions keyed by project name.

    Supports multiple providers (Claude, Codex, etc.) via LlmSessionFactory.
    Sessions are cached per project and recreated if provider changes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, LlmSession] = {}

    async def get_or_create_session(
        self,
        project_name: str,
        project_path: str,
        provider_id: str = "anthropic",
        model: str | None = None,
        cli_agent: CliAgentDefinition | None = None,
    ) -> LlmSession:
        """Get existing session or create one for the specified provider.

        If the provider changes, the existing session is closed and a new one created.
        """
        from llm.llm_session import LlmSessionFactory

        existing = self._sessions.get(project_name)

        # Close cached session when provider or working directory changes.
        # Without the path check, a task that targets a real repo (e.g. the
        # Code Bridge checkout) reuses the chat session that was bound to
        # ~/.code-bridge/global_chat, so every bash call runs in the wrong
        # cwd and `git log` reports "not a git repository".
        #
        # The cli_agent check is the same class of bug with sharper teeth: the
        # scope key is the task, so two steps of one task share a session. If a
        # definition-backed step reused the session a plain step opened, the
        # declared tools would never be applied; the other way round, a plain
        # step would silently inherit another agent's restrictions. A changed
        # source file also lands here, because the definition is compared by
        # value — the next run gets a session built from the new file.
        if existing is not None and (
            existing.provider_id != provider_id
            or getattr(existing, "project_path", None) != project_path
            or getattr(existing, "cli_agent", None) != cli_agent
        ):
            await existing.close()
            existing = None
            self._sessions.pop(project_name, None)

        if existing is None:
            import logging as _lg
            _lg.getLogger("llm.session_manager").warning(
                "[session_manager] creating session provider=%s project=%s path=%r model=%s cli_agent=%s",
                provider_id, project_name, project_path, model,
                cli_agent.name if cli_agent is not None else None,
            )
            session = LlmSessionFactory.create_session(
                provider_id=provider_id,
                project_path=project_path,
                model=model,
                cli_agent=cli_agent,
            )
            self._sessions[project_name] = session
        else:
            session = existing

        await session.set_model(model)
        return session

    def get_session_if_exists(self, project_name: str) -> LlmSession | None:
        """Return the cached session for this project, or None."""
        return self._sessions.get(project_name)

    async def close_session(self, project_name: str) -> None:
        """Close and remove one project session."""
        session = self._sessions.pop(project_name, None)
        if session is not None:
            await session.close()

    async def close_all(self) -> None:
        """Close and remove all sessions."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            await session.close()


_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
