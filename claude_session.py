"""Claude Code session management via SDK WebSocket transport."""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServer, WebSocketServerProtocol, serve


@dataclass
class ClaudeSession:
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
    def is_running(self) -> bool:
        """Whether Claude process is alive."""
        return self._process is not None and self._process.returncode is None

    @property
    def session_id(self) -> str | None:
        """Claude conversation session id."""
        return self._session_id

    @property
    def has_pending_permission_denials(self) -> bool:
        """Backward-compatible flag for pending permission request."""
        return self._pending_permission_request is not None

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

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_path,
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
        except Exception:
            pass

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
        """Start a new Claude turn."""
        if not message.strip():
            yield {"type": "error", "error": {"message": "Message content is empty"}}
            return

        try:
            if permission_mode and permission_mode.strip() and permission_mode != self.default_permission_mode:
                await self.close()
                self.default_permission_mode = permission_mode

            await self._ensure_process()

            if self._turn_in_progress:
                yield {"type": "error", "error": {"message": "Another Claude turn is already in progress"}}
                return

            async with self._turn_lock:
                self._turn_in_progress = True
                self._pending_permission_request = None

                await self._send_sdk_payload(
                    {
                        "type": "user",
                        "session_id": self._session_id or "",
                        "message": {"role": "user", "content": message},
                        "parent_tool_use_id": None,
                        "uuid": str(uuid.uuid4()),
                    }
                )

                async for event in self._stream_until_pause_or_result():
                    yield event
        except Exception as exc:
            self._turn_in_progress = False
            yield {"type": "error", "error": {"message": str(exc)}}

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
        except Exception as exc:
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
            except Exception:
                pass
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
    """Manage Claude sessions keyed by project name."""

    def __init__(self) -> None:
        self._sessions: dict[str, ClaudeSession] = {}

    async def get_or_create_session(
        self,
        project_name: str,
        project_path: str,
        model: str | None = None,
    ) -> ClaudeSession:
        """Get existing session or create one."""
        if project_name not in self._sessions:
            self._sessions[project_name] = ClaudeSession(project_path=project_path, model=model)
        session = self._sessions[project_name]
        await session.set_model(model)
        return session

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
