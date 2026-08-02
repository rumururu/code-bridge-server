"""Google Antigravity CLI session via `agy --print --output-format stream-json`.

The CLI streams a different envelope from the others: the discriminator is
``event`` rather than ``type``, assistant text arrives as ``text_delta`` on a
``step_update`` rather than as a message, and the closing frame nests its
answer at ``result.response``. Everything below exists to turn that into the
event shape the rest of Code Bridge already speaks, so a caller does not have
to know which CLI produced a turn.
"""

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

_CANDIDATE_PATHS = (
    "/opt/homebrew/bin/agy",
    "/usr/local/bin/agy",
    os.path.expanduser("~/.npm-global/bin/agy"),
)


@dataclass
class AntigravitySession(LlmSession):
    """Manage Antigravity CLI turns."""

    project_path: str
    model: str | None = None
    _agy_path: str = field(default="", init=False)
    _session_id: str | None = field(default=None, init=False)
    _process: asyncio.subprocess.Process | None = field(default=None, init=False)
    _stdout_task: asyncio.Task[None] | None = field(default=None, init=False)
    _stderr_task: asyncio.Task[None] | None = field(default=None, init=False)
    _event_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=asyncio.Queue, init=False
    )
    _turn_in_progress: bool = field(default=False, init=False)
    _stderr_lines: list[str] = field(default_factory=list, init=False)
    _full_response_text: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._agy_path = shutil.which("agy") or ""
        if not self._agy_path:
            for path in _CANDIDATE_PATHS:
                if os.path.exists(path):
                    self._agy_path = path
                    break

    @property
    def provider_id(self) -> str:
        return "antigravity"

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
            self._agy_path,
            "--print",
            message,
            "--output-format",
            "stream-json",
        ]
        if isinstance(self.model, str) and self.model.strip():
            cmd.extend(["--model", self.model.strip()])
        # An unattended turn has nobody to answer a permission prompt, and the
        # CLI's default is to stop and ask. Continuing the previous
        # conversation is what makes a multi-turn builder session coherent.
        if self._session_id:
            cmd.extend(["--conversation", self._session_id])
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
                if text:
                    await self._handle_jsonl_line(text)
        except asyncio.CancelledError:
            raise
        except (OSError, IOError, UnicodeDecodeError, ValueError, RuntimeError) as exc:
            logger.warning("Unexpected error reading Antigravity stdout: %s", exc)
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
            logger.warning("Unexpected error reading Antigravity stderr: %s", exc)

    async def _handle_jsonl_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            await self._event_queue.put({"type": "output", "text": line})
            return
        if not isinstance(event, dict):
            return

        normalized = self._normalize_event(event)
        if not normalized:
            return
        normalized["raw_event"] = event
        normalized["provider_id"] = self.provider_id
        normalized["provider"] = self.provider_id
        if self._session_id:
            normalized["session_id"] = self._session_id
        await self._event_queue.put(normalized)

    def _normalize_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        kind = str(event.get("event") or "").lower()

        if kind == "init":
            init = event.get("init") if isinstance(event.get("init"), dict) else {}
            conversation_id = event.get("conversation_id") or init.get("conversation_id")
            if isinstance(conversation_id, str) and conversation_id.strip():
                self._session_id = conversation_id.strip()
            return None

        if kind == "step_update":
            step = event.get("step_update")
            if not isinstance(step, dict):
                return None
            conversation_id = step.get("conversation_id")
            if isinstance(conversation_id, str) and conversation_id.strip():
                self._session_id = conversation_id.strip()

            delta = step.get("text_delta")
            if not isinstance(delta, str) or not delta:
                return None
            self._full_response_text += delta
            return {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": delta}],
                },
            }

        if kind == "result":
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            conversation_id = result.get("conversation_id")
            if isinstance(conversation_id, str) and conversation_id.strip():
                self._session_id = conversation_id.strip()

            status = str(result.get("status") or "").upper()
            response = result.get("response")
            if isinstance(response, str) and response:
                # The closing frame repeats the whole answer. Prefer it over the
                # accumulated deltas, which can be missing a tail if a frame was
                # dropped mid-stream.
                self._full_response_text = response

            if status and status != "SUCCESS":
                message = (
                    result.get("error")
                    or self._full_response_text
                    or "\n".join(self._stderr_lines[-5:])
                    or f"Antigravity CLI returned {status}"
                )
                return {"type": "error", "error": {"message": str(message)}}

            payload: dict[str, Any] = {
                "type": "result",
                "result": self._full_response_text,
            }
            usage = result.get("usage")
            if isinstance(usage, dict):
                payload["usage"] = usage
            return payload

        if kind in ("error", "failure", "failed"):
            message = event.get("error") or event.get("message") or "Antigravity CLI error"
            return {"type": "error", "error": {"message": str(message)}}

        return None

    async def send_message(self, message: str) -> AsyncGenerator[dict[str, Any], None]:
        if not self._agy_path:
            yield {
                "type": "error",
                "error": {
                    "message": (
                        "Antigravity CLI (agy) not found. Install it, or pick a "
                        "different provider in Settings."
                    )
                },
            }
            return

        if self._turn_in_progress:
            yield {
                "type": "error",
                "error": {"message": "A turn is already in progress"},
            }
            return

        self._turn_in_progress = True
        self._stderr_lines = []
        self._full_response_text = ""
        while not self._event_queue.empty():
            self._event_queue.get_nowait()

        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._build_command(message),
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            self._turn_in_progress = False
            yield {"type": "error", "error": {"message": f"Failed to start agy: {exc}"}}
            return

        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

        try:
            while True:
                event = await self._event_queue.get()
                if event.get("type") == "stream_end":
                    break
                yield event

            returncode = await self._process.wait()
            if returncode != 0 and not self._full_response_text:
                # A non-zero exit with nothing said is the case that used to
                # surface as an empty reply the user had to guess about.
                detail = "\n".join(self._stderr_lines[-5:]) or f"agy exited with {returncode}"
                yield {"type": "error", "error": {"message": detail}}
        finally:
            self._turn_in_progress = False
            for task in (self._stdout_task, self._stderr_task):
                if task and not task.done():
                    task.cancel()
            self._stdout_task = None
            self._stderr_task = None

    async def approve_pending_permissions_and_retry(
        self,
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "type": "error",
            "error": {
                "message": (
                    "Antigravity print mode does not expose a pending approval "
                    "to approve."
                )
            },
        }

    async def deny_pending_permissions(
        self,
        message: str = "Permission denied by user.",
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "type": "error",
            "error": {
                "message": (
                    "Antigravity print mode does not expose a pending approval "
                    "to deny."
                )
            },
        }

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
        except (OSError, ProcessLookupError):
            self._turn_in_progress = False
            return False

    async def set_model(self, model: str | None) -> None:
        next_model = model.strip() if isinstance(model, str) and model.strip() else None
        current = self.model.strip() if isinstance(self.model, str) and self.model.strip() else None
        if next_model == current:
            return
        self.model = next_model
        if self.is_running:
            await self.close()

    async def interrupt(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        try:
            process.send_signal(signal.SIGINT)
        except (OSError, ProcessLookupError):
            pass

    async def close(self) -> None:
        await self.interrupt()
        process = self._process
        if process is not None and process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except (OSError, ProcessLookupError):
                pass
        self._process = None
