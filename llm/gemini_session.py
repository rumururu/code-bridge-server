"""Google Gemini CLI session management via headless stream-json mode."""

import asyncio
import json
import logging
import os
import shutil
import signal
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from .llm_session import LlmSession

logger = logging.getLogger(__name__)


@dataclass
class GeminiSession(LlmSession):
    """Manage Gemini CLI turns using `gemini -p ... --output-format stream-json`."""

    project_path: str
    model: str | None = None
    _gemini_path: str = field(default="", init=False)
    _session_id: str | None = field(default=None, init=False)
    _process: asyncio.subprocess.Process | None = field(default=None, init=False)
    _stdout_task: asyncio.Task[None] | None = field(default=None, init=False)
    _stderr_task: asyncio.Task[None] | None = field(default=None, init=False)
    _event_queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue, init=False)
    _turn_in_progress: bool = field(default=False, init=False)
    _stderr_lines: list[str] = field(default_factory=list, init=False)
    _full_response_text: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._gemini_path = shutil.which("gemini") or ""
        if not self._gemini_path:
            for path in (
                os.path.expanduser("~/.npm-global/bin/gemini"),
                "/usr/local/bin/gemini",
                "/opt/homebrew/bin/gemini",
            ):
                if os.path.exists(path):
                    self._gemini_path = path
                    break

    @property
    def provider_id(self) -> str:
        return "google"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def has_pending_permission_denials(self) -> bool:
        return False

    def _build_command(self, message: str) -> list[str]:
        cmd = [
            self._gemini_path,
            "-p",
            message,
            "--output-format",
            "stream-json",
        ]
        if isinstance(self.model, str) and self.model.strip():
            cmd.extend(["-m", self.model.strip()])
        return cmd

    async def _read_stdout(self) -> None:
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
        except asyncio.CancelledError:
            raise
        except (OSError, IOError, UnicodeDecodeError, ValueError, RuntimeError) as exc:
            logger.warning("Unexpected error reading Gemini stdout: %s", exc)
        finally:
            await self._event_queue.put({"type": "stream_end"})

    async def _read_stderr(self) -> None:
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
        except asyncio.CancelledError:
            raise
        except (OSError, IOError, UnicodeDecodeError, ValueError, RuntimeError) as exc:
            logger.warning("Unexpected error reading Gemini stderr: %s", exc)

    async def _handle_jsonl_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            await self._event_queue.put({"type": "output", "text": line})
            return

        if not isinstance(event, dict):
            return

        session_id = event.get("session_id") or event.get("id")
        if isinstance(session_id, str) and session_id.strip():
            self._session_id = session_id

        normalized = self._normalize_event(event)
        if normalized:
            normalized["raw_event"] = event
            normalized["provider_id"] = self.provider_id
            normalized["provider"] = self.provider_id
            if self._session_id:
                normalized["session_id"] = self._session_id
            await self._event_queue.put(normalized)

    def _normalize_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = str(event.get("type", "") or "").lower()
        text = self._extract_text(event)

        if event_type in ("error", "failed", "failure"):
            return {
                "type": "error",
                "error": {"message": text or str(event.get("error") or "Gemini CLI error")},
            }

        if "tool" in event_type and "result" not in event_type:
            tool_name = event.get("name") or event.get("tool_name") or event.get("tool")
            if tool_name:
                # Gemini CLI emits the tool call id under ``id`` (preferred)
                # with ``tool_use_id`` as a rarer fallback. Mirror the codex
                # session pattern (TASK_001) so a single key ordering applies
                # across providers and AgentTaskRunSink can persist it.
                raw_call_id = event.get("id") or event.get("tool_use_id")
                call_id_str = str(raw_call_id) if raw_call_id else None
                return {
                    "type": "assistant",
                    "call_id": call_id_str,
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": call_id_str,
                            "name": tool_name,
                            "input": event.get("input") if isinstance(event.get("input"), dict) else {},
                        }],
                    },
                }

        if "tool" in event_type and "result" in event_type:
            # tool_result side prefers ``tool_use_id`` (Anthropic-style) and
            # falls back to ``id``. Same fallback order as TASK_001 codex.
            raw_call_id = event.get("tool_use_id") or event.get("id")
            call_id_str = str(raw_call_id) if raw_call_id else None
            return {
                "type": "assistant",
                "call_id": call_id_str,
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": call_id_str,
                        "content": text or str(event.get("result") or event.get("output") or ""),
                        "is_error": bool(event.get("is_error", False)),
                    }],
                },
            }

        if event_type in ("done", "complete", "completed", "result", "response.completed"):
            if text:
                self._full_response_text += text
            result: dict[str, Any] = {"type": "result", "result": self._full_response_text}
            usage = event.get("usage") or event.get("stats")
            if isinstance(usage, dict):
                result["usage"] = usage
            return result

        if text and event_type in (
            "",
            "content",
            "message",
            "text",
            "output",
            "response",
            "assistant",
            "response.output_text.delta",
        ):
            self._full_response_text += text
            return {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            }

        if event_type in ("status", "progress", "thinking"):
            return {"type": "output", "text": text or str(event)}

        return {"type": "provider_event", "event": event, "legacy_type": "gemini_event"}

    def _extract_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(self._extract_text(item) for item in value)
        if not isinstance(value, dict):
            return ""

        for key in ("text", "content", "delta", "message", "response", "output", "result"):
            nested = value.get(key)
            if isinstance(nested, str) and nested:
                return nested
            if isinstance(nested, (dict, list)):
                text = self._extract_text(nested)
                if text:
                    return text
        return ""

    async def send_message(
        self,
        message: str,
        permission_mode: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if not message.strip():
            yield {"type": "error", "error": {"message": "Message content is empty"}}
            return

        if not self._gemini_path:
            yield {
                "type": "error",
                "error": {
                    "message": "Gemini CLI not found. Install with: npm install -g @google/gemini-cli",
                },
            }
            return

        try:
            self._turn_in_progress = True
            self._stderr_lines.clear()
            self._event_queue = asyncio.Queue()
            self._full_response_text = ""

            self._process = await asyncio.create_subprocess_exec(
                *self._build_command(message),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env=os.environ.copy(),
            )

            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            while True:
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=300.0)
                except asyncio.TimeoutError:
                    yield {"type": "error", "error": {"message": "Gemini response timed out"}}
                    break

                event_type = event.get("type")
                if event_type == "stream_end":
                    if self._process:
                        await self._process.wait()
                    if self._full_response_text:
                        yield {"type": "result", "result": self._full_response_text}
                    elif self._stderr_lines:
                        yield {
                            "type": "error",
                            "error": {"message": "\n".join(self._stderr_lines[-5:])},
                        }
                    break

                if event_type in ("result", "error"):
                    yield event
                    break

                yield event
        except (OSError, ConnectionError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
            yield {"type": "error", "error": {"message": str(exc)}}
        finally:
            self._turn_in_progress = False
            await self._cleanup_process()

    async def _cleanup_process(self) -> None:
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
        yield {
            "type": "error",
            "error": {"message": "Gemini headless mode does not expose a pending approval to approve."},
        }

    async def deny_pending_permissions(
        self,
        message: str = "Permission denied by user.",
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "type": "error",
            "error": {"message": "Gemini headless mode does not expose a pending approval to deny."},
        }

    async def close(self) -> None:
        await self._cleanup_process()
        self._turn_in_progress = False

    async def abort_current_turn(self) -> bool:
        if not self._turn_in_progress:
            return False

        process = self._process
        if process is None or process.returncode is not None:
            self._turn_in_progress = False
            return False

        try:
            process.send_signal(signal.SIGINT)
            self._turn_in_progress = False
            return True
        except ProcessLookupError:
            self._turn_in_progress = False
            return False

    async def set_model(self, model: str | None) -> None:
        next_model = model.strip() if isinstance(model, str) and model.strip() else None
        current_model = self.model.strip() if isinstance(self.model, str) and self.model.strip() else None
        if next_model == current_model:
            return
        self.model = next_model
        if self.is_running:
            await self.close()
