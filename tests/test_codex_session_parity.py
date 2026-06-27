import asyncio
import json
import signal
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm import codex_session
from llm.codex_session import CodexSession


class _FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    async def readline(self):
        await asyncio.sleep(0)
        if not self._lines:
            return b""
        line = self._lines.pop(0)
        if isinstance(line, bytes):
            return line
        return (line + "\n").encode("utf-8")


class _FakeStdin:
    def __init__(self):
        self.writes = []
        self.closed = False
        self.drained = False

    def write(self, data):
        self.writes.append(data)

    async def drain(self):
        self.drained = True

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, stdout_lines=(), stderr_lines=(), stdin=None, returncode=0):
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines)
        self.stdin = stdin
        self.returncode = returncode
        self.signals = []
        self.killed = False
        self.waited = False

    async def wait(self):
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        if sig in (signal.SIGINT, signal.SIGTERM):
            self.returncode = -sig

    def kill(self):
        self.killed = True
        self.returncode = -9


def _jsonl(event):
    return json.dumps(event)


async def _collect_events(session, message):
    return [event async for event in session.send_message(message)]


class CodexSessionParityTest(unittest.IsolatedAsyncioTestCase):
    def _session(self, *, model="gpt-5", sandbox_mode="workspace-write"):
        session = CodexSession(
            project_path="/tmp/code-bridge",
            model=model,
            sandbox_mode=sandbox_mode,
        )
        session._codex_path = "/usr/local/bin/codex"
        return session

    async def test_new_turn_builds_codex_exec_command_without_running_binary(self):
        session = self._session(model=" gpt-5 ")
        calls = []
        process = _FakeProcess(
            stdout_lines=[
                _jsonl({"type": "item.completed", "session_id": "sess-new", "item": {"type": "agent_message", "text": "hello"}}),
                _jsonl({"type": "turn.completed", "usage": {"input_tokens": 3}}),
            ],
        )

        async def fake_create_subprocess_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return process

        with patch.object(codex_session.asyncio, "create_subprocess_exec", fake_create_subprocess_exec):
            events = await _collect_events(session, "write tests")

        self.assertEqual(
            list(calls[0][0]),
            [
                "/usr/local/bin/codex",
                "exec",
                "--json",
                "--skip-git-repo-check",
                "-C",
                "/tmp/code-bridge",
                "-s",
                "workspace-write",
                "-m",
                "gpt-5",
                "write tests",
            ],
        )
        self.assertEqual(calls[0][1]["stdin"], asyncio.subprocess.DEVNULL)
        self.assertEqual(calls[0][1]["stdout"], asyncio.subprocess.PIPE)
        self.assertEqual(calls[0][1]["stderr"], asyncio.subprocess.PIPE)
        self.assertEqual(calls[0][1]["cwd"], "/tmp/code-bridge")
        self.assertEqual(session.session_id, "sess-new")
        self.assertEqual(events[0]["type"], "assistant")
        self.assertEqual(events[0]["message"]["content"][0]["text"], "hello")
        self.assertEqual(events[0]["raw_event"]["type"], "item.completed")
        self.assertEqual(events[0]["provider_id"], "openai")
        self.assertEqual(events[1]["type"], "result")
        self.assertEqual(events[1]["result"], "hello")
        self.assertEqual(events[1]["usage"], {"input_tokens": 3})
        self.assertEqual(events[1]["raw_event"]["type"], "turn.completed")

    async def test_resume_turn_uses_native_resume_command_with_prompt_argument(self):
        session = self._session(model="gpt-5")
        session._session_id = "sess-existing"
        calls = []
        process = _FakeProcess(
            stdout_lines=[_jsonl({"type": "turn.completed"})],
        )

        async def fake_create_subprocess_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return process

        with patch.object(codex_session.asyncio, "create_subprocess_exec", fake_create_subprocess_exec):
            events = await _collect_events(session, "continue the turn")

        self.assertEqual(
            list(calls[0][0]),
            [
                "/usr/local/bin/codex",
                "exec",
                "resume",
                "sess-existing",
                "--json",
                "--skip-git-repo-check",
                "-m",
                "gpt-5",
                "continue the turn",
            ],
        )
        self.assertEqual(calls[0][1]["stdin"], asyncio.subprocess.DEVNULL)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "result")
        self.assertEqual(events[0]["result"], "")
        self.assertEqual(events[0]["raw_event"]["type"], "turn.completed")

    async def test_resume_session_pins_session_id_and_closes_running_process(self):
        session = self._session()
        session._process = _FakeProcess(returncode=None)

        await session.resume_session(" sess-resume ")

        self.assertEqual(session.session_id, "sess-resume")
        self.assertIsNone(session._process)

    async def test_item_completed_and_turn_completed_normalize_codex_json_schema(self):
        session = self._session(model=None)

        await session._handle_jsonl_line(
            _jsonl(
                {
                    "type": "item.completed",
                    "session_id": "sess-jsonl",
                    "item": {
                        "type": "agent_message",
                        "content": [{"type": "output_text", "text": "first"}, {"text": " second"}],
                    },
                }
            )
        )
        await session._handle_jsonl_line(
            _jsonl({"type": "turn.completed", "usage": {"output_tokens": 2}})
        )

        assistant_event = await session._event_queue.get()
        result_event = await session._event_queue.get()

        self.assertEqual(session.session_id, "sess-jsonl")
        self.assertEqual(assistant_event["type"], "assistant")
        self.assertEqual(assistant_event["message"]["content"][0]["text"], "first second")
        self.assertEqual(assistant_event["raw_event"]["type"], "item.completed")
        self.assertEqual(result_event["type"], "result")
        self.assertEqual(result_event["result"], "first second")
        self.assertEqual(result_event["usage"], {"output_tokens": 2})
        self.assertEqual(result_event["raw_event"]["type"], "turn.completed")

    async def test_reasoning_item_completed_normalizes_to_output(self):
        session = self._session()

        await session._handle_jsonl_line(
            _jsonl({"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}})
        )

        event = await session._event_queue.get()
        self.assertEqual(event["type"], "output")
        self.assertEqual(event["text"], "thinking")
        self.assertEqual(event["raw_event"]["type"], "item.completed")

    async def test_abort_current_turn_sends_sigint_and_clears_turn_state(self):
        session = self._session()
        process = _FakeProcess(returncode=None)
        session._process = process
        session._turn_in_progress = True
        session._pending_permission_request = {"id": "perm"}

        aborted = await session.abort_current_turn()

        self.assertTrue(aborted)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertFalse(session._turn_in_progress)
        self.assertIsNone(session._pending_permission_request)

    async def test_abort_current_turn_returns_false_without_active_process(self):
        session = self._session()

        self.assertFalse(await session.abort_current_turn())

        session._turn_in_progress = True
        session._process = _FakeProcess(returncode=0)

        self.assertFalse(await session.abort_current_turn())
        self.assertFalse(session._turn_in_progress)
