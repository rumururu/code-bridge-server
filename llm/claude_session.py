"""Claude Code session management via SDK WebSocket transport."""

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServer, WebSocketServerProtocol, serve

from .llm_session import LlmSession

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


@dataclass
class ClaudeSession(LlmSession):
    """Manage one long-lived Claude CLI process with real-time control responses."""

    project_path: str
    default_permission_mode: str = "default"
    model: str | None = None
    _claude_path: str = field(default="", init=False)
    _session_id: str | None = field(default=None, init=False)
    _process: asyncio.subprocess.Process | None = field(default=None, init=False)
    _stderr_task: asyncio.Task[None] | None = field(default=None, init=False)
    _process_wait_task: asyncio.Task[None] | None = field(default=None, init=False)
    _event_queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue, init=False)
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _turn_in_progress: bool = field(default=False, init=False)
    _pending_permission_request: dict[str, Any] | None = field(default=None, init=False)
    _stderr_lines: list[str] = field(default_factory=list, init=False)

    # Internal SDK transport server state.
    _sdk_server: WebSocketServer | None = field(default=None, init=False)
    _sdk_url: str | None = field(default=None, init=False)
    _sdk_token: str | None = field(default=None, init=False)
    _sdk_connection: WebSocketServerProtocol | None = field(default=None, init=False)
    _sdk_connected_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _sdk_send_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue, init=False)
    _sdk_send_task: asyncio.Task[None] | None = field(default=None, init=False)

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
        """Whether Claude process is alive."""
        return self._process is not None and self._process.returncode is None

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

    def _build_command(self, permission_mode: str | None = None) -> list[str]:
        """Build Claude command in SDK mode."""
        if not self._sdk_url:
            raise RuntimeError("SDK URL is not initialized")

        cmd = [
            self._claude_path,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--include-partial-messages",
            "--sdk-url",
            self._sdk_url,
            "--append-system-prompt",
            ASK_USER_SYSTEM_PROMPT,
        ]

        if self._session_id:
            cmd.extend(["--resume", self._session_id])

        if isinstance(self.model, str) and self.model.strip():
            cmd.extend(["--model", self.model.strip()])

        resolved_mode = (permission_mode or self.default_permission_mode).strip()
        if resolved_mode:
            cmd.extend(["--permission-mode", resolved_mode])

        return cmd

    @staticmethod
    def _normalize_line(line: bytes) -> str:
        return line.decode("utf-8", errors="replace").strip()

    async def _ensure_sdk_server(self) -> None:
        """Start local WS server that Claude CLI uses for stream-json transport."""
        if self._sdk_server is not None:
            return

        self._sdk_token = uuid.uuid4().hex
        self._sdk_connected_event.clear()
        self._sdk_send_queue = asyncio.Queue()

        async def _handler(websocket: WebSocketServerProtocol, path: str) -> None:
            token = path.lstrip("/")
            if token != self._sdk_token:
                await websocket.close(code=1008, reason="Invalid token")
                return

            if self._sdk_connection is not None:
                await websocket.close(code=1013, reason="Session already connected")
                return

            self._sdk_connection = websocket
            self._sdk_connected_event.set()
            self._sdk_send_task = asyncio.create_task(self._sdk_sender_loop(websocket))

            try:
                async for raw_payload in websocket:
                    payload = raw_payload.decode("utf-8", errors="replace") if isinstance(raw_payload, bytes) else str(raw_payload)
                    for line in payload.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        await self._handle_sdk_line(line)
            except ConnectionClosed:
                pass
            finally:
                if self._sdk_send_task and not self._sdk_send_task.done():
                    self._sdk_send_task.cancel()
                    try:
                        await self._sdk_send_task
                    except asyncio.CancelledError:
                        pass
                self._sdk_send_task = None
                self._sdk_connection = None
                self._sdk_connected_event.clear()

        self._sdk_server = await serve(_handler, "127.0.0.1", 0)
        socket = self._sdk_server.sockets[0]
        port = socket.getsockname()[1]
        self._sdk_url = f"ws://127.0.0.1:{port}/{self._sdk_token}"

    async def _sdk_sender_loop(self, websocket: WebSocketServerProtocol) -> None:
        """Send queued JSON-lines to Claude SDK WebSocket."""
        while True:
            line = await self._sdk_send_queue.get()
            await websocket.send(line)

    async def _handle_sdk_line(self, line: str) -> None:
        """Parse one SDK transport line and enqueue normalized event."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            await self._event_queue.put({"type": "output", "text": line})
            return

        if not isinstance(event, dict):
            return

        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            self._session_id = session_id

        await self._event_queue.put(event)

    async def _ensure_process(self) -> None:
        """Start Claude process if missing/dead."""
        if not self._claude_path:
            raise RuntimeError("Claude CLI not found")

        async with self._start_lock:
            if self.is_running:
                return

            self._turn_in_progress = False
            self._pending_permission_request = None
            self._stderr_lines.clear()
            self._event_queue = asyncio.Queue()

            await self._ensure_sdk_server()
            cmd = self._build_command()

            # Resolve a safe cwd. When the stored project path is missing we
            # fall back to the user's home so Claude CLI can still spawn, and
            # remember a note to pass along on the next user message.
            cwd, fallback_note = self._resolve_start_cwd()
            if fallback_note is not None:
                self._fallback_note = fallback_note
                self._fallback_note_delivered = False
            import logging as _lg
            _lg.getLogger("llm.claude_session").warning(
                "[claude_session] start project_path=%r cwd=%r fallback=%s",
                self.project_path,
                cwd,
                "yes" if fallback_note else "no",
            )

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
            )

            self._stderr_task = asyncio.create_task(self._read_stderr())
            self._process_wait_task = asyncio.create_task(self._wait_for_process_exit())

            try:
                await asyncio.wait_for(self._sdk_connected_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                await self.close()
                raise RuntimeError("Claude SDK transport connection timed out")

    async def _read_stderr(self) -> None:
        """Read stderr for diagnostics."""
        process = self._process
        if process is None or process.stderr is None:
            return

        try:
            while True:
                raw_line = await process.stderr.readline()
                if not raw_line:
                    break
                text = self._normalize_line(raw_line)
                if text:
                    self._stderr_lines.append(text)
        except asyncio.CancelledError:
            raise
        except (OSError, IOError):
            # Process terminated or pipe closed - expected during shutdown
            pass
        except (UnicodeDecodeError, ValueError, RuntimeError) as exc:
            logger.warning("Unexpected error reading stderr: %s", exc)

    async def _wait_for_process_exit(self) -> None:
        """Watch process exit and emit session_closed event."""
        process = self._process
        if process is None:
            return
        returncode = await process.wait()
        stderr_text = "\n".join(self._stderr_lines).strip()
        await self._event_queue.put(
            {
                "type": "session_closed",
                "returncode": returncode,
                "stderr": stderr_text,
            }
        )

    async def _send_sdk_payload(self, payload: dict[str, Any]) -> None:
        """Send one JSON payload to Claude through SDK websocket."""
        if not self.is_running:
            raise RuntimeError("Claude session process is not running")

        if self._sdk_connection is None:
            await asyncio.wait_for(self._sdk_connected_event.wait(), timeout=10.0)
            if self._sdk_connection is None:
                raise RuntimeError("Claude SDK transport is not connected")

        line = json.dumps(payload, ensure_ascii=False) + "\n"
        await self._sdk_send_queue.put(line)

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

            yield event

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

            with open("/tmp/cb_debug.log", "a") as f:
                f.write(f"[claude_session] send_message enter path={self.project_path!r} running={self.is_running}\n")
            await self._ensure_process()
            with open("/tmp/cb_debug.log", "a") as f:
                f.write(f"[claude_session] _ensure_process returned running={self.is_running}\n")

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

                await self._send_sdk_payload(
                    {
                        "type": "user",
                        "session_id": self._session_id or "",
                        "message": {"role": "user", "content": content},
                        "parent_tool_use_id": None,
                        "uuid": str(uuid.uuid4()),
                    }
                )

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

        request_id = pending.get("request_id")
        request = pending.get("request", {})
        if not isinstance(request_id, str) or not isinstance(request, dict):
            yield {"type": "error", "error": {"message": "Invalid pending permission request state"}}
            return

        tool_input = request.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}

        tool_use_id = request.get("tool_use_id")
        tool_use_id_value = tool_use_id if isinstance(tool_use_id, str) else None

        if allow:
            response_payload: dict[str, Any] = {"behavior": "allow", "updatedInput": tool_input}
            if tool_use_id_value:
                response_payload["toolUseID"] = tool_use_id_value
        else:
            response_payload = {"behavior": "deny", "message": deny_message, "interrupt": False}
            if tool_use_id_value:
                response_payload["toolUseID"] = tool_use_id_value

        try:
            await self._send_sdk_payload(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": request_id,
                        "response": response_payload,
                    },
                }
            )
        except (RuntimeError, OSError) as exc:
            self._turn_in_progress = False
            self._pending_permission_request = None
            yield {"type": "error", "error": {"message": str(exc)}}
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
        """Close process and internal SDK transport server."""
        process = self._process
        if process and process.returncode is None:
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

        for task in (self._stderr_task, self._process_wait_task, self._sdk_send_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._sdk_connection is not None:
            try:
                await self._sdk_connection.close()
            except ConnectionClosed:
                # Already closed - expected
                pass
            except (OSError, RuntimeError, ConnectionError) as exc:
                logger.debug("Error closing SDK connection: %s", exc)
            self._sdk_connection = None
        self._sdk_connected_event.clear()

        if self._sdk_server is not None:
            self._sdk_server.close()
            await self._sdk_server.wait_closed()
            self._sdk_server = None
        self._sdk_url = None
        self._sdk_token = None

        self._process = None
        self._stderr_task = None
        self._process_wait_task = None
        self._sdk_send_task = None
        self._turn_in_progress = False
        self._pending_permission_request = None

    async def abort_current_turn(self) -> bool:
        """Abort the current turn by sending SIGINT to the Claude process.

        Returns True if abort signal was sent, False if no turn in progress.
        """
        if not self._turn_in_progress:
            return False

        process = self._process
        if process is None or process.returncode is not None:
            self._turn_in_progress = False
            return False

        try:
            process.send_signal(signal.SIGINT)
            self._turn_in_progress = False
            self._pending_permission_request = None
            return True
        except ProcessLookupError:
            self._turn_in_progress = False
            return False

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
        if self.is_running:
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
    ) -> LlmSession:
        """Get existing session or create one for the specified provider.

        If the provider changes, the existing session is closed and a new one created.
        """
        from llm.llm_session import LlmSessionFactory

        existing = self._sessions.get(project_name)

        # If provider changed, close old session
        if existing is not None and existing.provider_id != provider_id:
            await existing.close()
            existing = None
            self._sessions.pop(project_name, None)

        if existing is None:
            import logging as _lg
            _lg.getLogger("llm.session_manager").warning(
                "[session_manager] creating session provider=%s project=%s path=%r model=%s",
                provider_id, project_name, project_path, model,
            )
            session = LlmSessionFactory.create_session(
                provider_id=provider_id,
                project_path=project_path,
                model=model,
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
