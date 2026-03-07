"""OpenAI Codex session management via CLI exec mode.

Uses `codex exec --json` for non-interactive execution with JSONL output streaming.
Session persistence is handled via Codex's built-in session files for resume capability.
"""

import asyncio
import json
import os
import shutil
import signal
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from llm_session import LlmSession


@dataclass
class CodexSession(LlmSession):
    """Manage OpenAI Codex CLI sessions with JSONL streaming.

    Unlike Claude's SDK WebSocket mode, Codex uses a simpler exec mode where
    each message spawns a new process. Session continuity is maintained via
    Codex's built-in session resume feature.
    """

    project_path: str
    model: str | None = None
    sandbox_mode: str | None = None  # read-only, workspace-write, danger-full-access (loaded from settings)
    _codex_path: str = field(default="", init=False)
    _session_id: str | None = field(default=None, init=False)
    _process: asyncio.subprocess.Process | None = field(default=None, init=False)
    _stdout_task: asyncio.Task[None] | None = field(default=None, init=False)
    _stderr_task: asyncio.Task[None] | None = field(default=None, init=False)
    _event_queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue, init=False)
    _turn_in_progress: bool = field(default=False, init=False)
    _pending_permission_request: dict[str, Any] | None = field(default=None, init=False)
    _stderr_lines: list[str] = field(default_factory=list, init=False)
    _full_response_text: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Locate Codex executable and load settings."""
        # Load sandbox mode from settings if not provided
        if self.sandbox_mode is None:
            from llm_settings import get_codex_sandbox_mode
            self.sandbox_mode = get_codex_sandbox_mode()

        self._codex_path = shutil.which("codex") or ""
        if not self._codex_path:
            # Check common npm global bin paths
            for path in (
                os.path.expanduser("~/.npm-global/bin/codex"),
                "/usr/local/bin/codex",
                "/opt/homebrew/bin/codex",
            ):
                if os.path.exists(path):
                    self._codex_path = path
                    break

    @property
    def provider_id(self) -> str:
        """Return the provider identifier."""
        return "openai"

    @property
    def is_running(self) -> bool:
        """Whether Codex process is alive."""
        return self._process is not None and self._process.returncode is None

    @property
    def session_id(self) -> str | None:
        """Codex session ID for resume capability."""
        return self._session_id

    @property
    def has_pending_permission_denials(self) -> bool:
        """Codex uses --full-auto, so no pending permissions in normal operation."""
        return self._pending_permission_request is not None

    def _build_command(self, message: str, is_resume: bool = False) -> list[str]:
        """Build Codex exec command."""
        if is_resume and self._session_id:
            # Resume previous session
            cmd = [
                self._codex_path,
                "exec",
                "resume",
                self._session_id,
                "--json",
                "--full-auto",  # Auto-approve in sandboxed mode
                "-C", self.project_path,
            ]
        else:
            # New session
            cmd = [
                self._codex_path,
                "exec",
                "--json",
                "--full-auto",  # Auto-approve in sandboxed mode
                "-s", self.sandbox_mode,
                "-C", self.project_path,
            ]

        if isinstance(self.model, str) and self.model.strip():
            cmd.extend(["-m", self.model.strip()])

        if not is_resume:
            # Add the prompt as the last argument for new sessions
            cmd.append(message)

        return cmd

    async def _read_stdout(self) -> None:
        """Read stdout JSONL events from Codex process."""
        process = self._process
        if process is None or process.stdout is None:
            return

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                await self._handle_jsonl_line(text)
        except Exception:
            pass
        finally:
            # Signal end of stream
            await self._event_queue.put({"type": "stream_end"})

    async def _read_stderr(self) -> None:
        """Read stderr for diagnostics."""
        process = self._process
        if process is None or process.stderr is None:
            return

        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self._stderr_lines.append(text)
        except Exception:
            pass

    async def _handle_jsonl_line(self, line: str) -> None:
        """Parse one JSONL line and enqueue normalized event."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON output, treat as status message
            await self._event_queue.put({"type": "output", "text": line})
            return

        if not isinstance(event, dict):
            return

        event_type = event.get("type", "")

        # Extract session ID from events
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            self._session_id = session_id

        # Normalize Codex events to match Claude event format
        normalized = self._normalize_event(event)
        if normalized:
            await self._event_queue.put(normalized)

    def _normalize_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize Codex event to Claude-compatible format."""
        event_type = event.get("type", "")

        # Message events
        if event_type == "message":
            message = event.get("message", {})
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "assistant":
                # Accumulate response text
                if isinstance(content, str):
                    self._full_response_text += content

                return {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": content}] if isinstance(content, str) else content,
                    },
                }

        # New Codex JSON schema: item.completed with item.type=agent_message
        if event_type == "item.completed":
            item = event.get("item", {})
            if not isinstance(item, dict):
                return {"type": "codex_event", "event": event}

            item_type = str(item.get("type", "") or "")
            text = self._extract_item_text(item)

            if item_type in ("agent_message", "assistant_message"):
                if text:
                    self._full_response_text += text
                    return {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        },
                    }
                return None

            if item_type in ("reasoning", "thinking"):
                if text:
                    return {"type": "output", "text": text}
                return None

        # Tool use events (Codex command execution)
        if event_type == "tool_use" or event_type == "function_call":
            return {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "name": event.get("name", event.get("function", "unknown")),
                        "input": event.get("input", event.get("arguments", {})),
                    }],
                },
            }

        # Tool result events
        if event_type == "tool_result" or event_type == "function_result":
            return {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_result",
                        "content": str(event.get("result", event.get("output", ""))),
                    }],
                },
            }

        # New Codex JSON schema: explicit turn completion marker
        if event_type == "turn.completed":
            normalized: dict[str, Any] = {
                "type": "result",
                "result": self._full_response_text,
            }
            usage = event.get("usage")
            if isinstance(usage, dict):
                normalized["usage"] = usage
            return normalized

        # Completion event
        if event_type in ("done", "complete", "finished"):
            return {
                "type": "result",
                "result": self._full_response_text,
            }

        # Error events
        if event_type == "error":
            return {
                "type": "error",
                "error": {"message": event.get("message", event.get("error", "Unknown error"))},
            }

        # Status/progress events
        if event_type in ("status", "progress", "thinking"):
            return {
                "type": "output",
                "text": event.get("message", event.get("text", str(event))),
            }

        # Pass through unknown events for debugging
        return {"type": "codex_event", "event": event}

    def _extract_item_text(self, item: dict[str, Any]) -> str:
        """Best-effort text extraction from Codex item payloads."""
        text = item.get("text")
        if isinstance(text, str) and text:
            return text

        content = item.get("content")
        if isinstance(content, str) and content:
            return content

        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text:
                    chunks.append(part_text)
            return "".join(chunks)

        return ""

    async def send_message(
        self,
        message: str,
        permission_mode: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute a Codex turn and stream response events."""
        if not message.strip():
            yield {"type": "error", "error": {"message": "Message content is empty"}}
            return

        if not self._codex_path:
            yield {"type": "error", "error": {"message": "Codex CLI not found. Install with: npm i -g @openai/codex"}}
            return

        try:
            # Reset state
            self._turn_in_progress = True
            self._pending_permission_request = None
            self._stderr_lines.clear()
            self._event_queue = asyncio.Queue()
            self._full_response_text = ""

            # Build command (resume if we have a session, otherwise new)
            is_resume = self._session_id is not None
            cmd = self._build_command(message, is_resume=is_resume)

            # For resume, we need to send the message via stdin
            stdin_mode = asyncio.subprocess.PIPE if is_resume else asyncio.subprocess.DEVNULL

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=stdin_mode,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env=os.environ.copy(),
            )

            # If resuming, write message to stdin
            if is_resume and self._process.stdin:
                self._process.stdin.write((message + "\n").encode("utf-8"))
                await self._process.stdin.drain()
                self._process.stdin.close()

            # Start reading stdout and stderr
            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            # Stream events until completion
            while True:
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=300.0)
                except asyncio.TimeoutError:
                    yield {"type": "error", "error": {"message": "Codex response timed out"}}
                    break

                event_type = event.get("type")

                if event_type == "stream_end":
                    # Process ended, wait for it
                    if self._process:
                        await self._process.wait()

                    # If we didn't get a result event, generate one
                    if self._full_response_text:
                        yield {"type": "result", "result": self._full_response_text}
                    elif self._stderr_lines:
                        yield {
                            "type": "error",
                            "error": {"message": "\n".join(self._stderr_lines[-5:])},
                        }
                    break

                if event_type == "result":
                    yield event
                    break

                if event_type == "error":
                    yield event
                    break

                yield event

        except Exception as exc:
            yield {"type": "error", "error": {"message": str(exc)}}
        finally:
            self._turn_in_progress = False
            await self._cleanup_process()

    async def _cleanup_process(self) -> None:
        """Clean up process and tasks."""
        for task in (self._stdout_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._process and self._process.returncode is None:
            try:
                self._process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass

        self._process = None
        self._stdout_task = None
        self._stderr_task = None

    async def approve_pending_permissions_and_retry(
        self,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Codex uses --full-auto mode, permissions are auto-approved."""
        yield {
            "type": "error",
            "error": {"message": "Codex runs in auto-approval mode. No manual approval needed."},
        }

    async def deny_pending_permissions(
        self,
        message: str = "Permission denied by user.",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Codex uses --full-auto mode, cannot deny permissions mid-turn."""
        yield {
            "type": "error",
            "error": {"message": "Codex runs in auto-approval mode. Cannot deny permissions."},
        }

    async def close(self) -> None:
        """Close session and clean up."""
        await self._cleanup_process()
        self._turn_in_progress = False
        self._pending_permission_request = None
        # Note: We keep _session_id for potential resume later

    async def abort_current_turn(self) -> bool:
        """Abort the current turn by sending SIGINT to the Codex process.

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
        """Set the model for subsequent turns."""
        next_model = model.strip() if isinstance(model, str) and model.strip() else None
        current_model = self.model.strip() if isinstance(self.model, str) and self.model.strip() else None

        if next_model == current_model:
            return

        self.model = next_model
        # Clear session when model changes to start fresh
        if self.is_running:
            await self.close()
        self._session_id = None
