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

from llm import gemini_session
from llm.gemini_session import GeminiSession


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


class _FakeProcess:
    def __init__(self, stdout_lines=(), stderr_lines=(), returncode=0):
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines)
        self.returncode = returncode
        self.signals = []
        self.killed = False

    async def wait(self):
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


class GeminiSessionParityTest(unittest.IsolatedAsyncioTestCase):
    def _session(self, *, model="gemini-2.5-pro"):
        session = GeminiSession(project_path="/tmp/code-bridge", model=model)
        session._gemini_path = "/usr/local/bin/gemini"
        return session

    async def test_new_turn_builds_gemini_stream_json_command(self):
        session = self._session(model=" gemini-2.5-flash ")
        calls = []
        process = _FakeProcess(
            stdout_lines=[
                _jsonl({"type": "content", "text": "hello "}),
                _jsonl({"type": "content", "text": "gemini"}),
                _jsonl({"type": "completed", "usage": {"input_tokens": 3}}),
            ],
        )

        async def fake_create_subprocess_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return process

        with patch.object(gemini_session.asyncio, "create_subprocess_exec", fake_create_subprocess_exec):
            events = await _collect_events(session, "write tests")

        self.assertEqual(
            list(calls[0][0]),
            [
                "/usr/local/bin/gemini",
                "-p",
                "write tests",
                "--output-format",
                "stream-json",
                "-m",
                "gemini-2.5-flash",
            ],
        )
        self.assertEqual(calls[0][1]["stdin"], asyncio.subprocess.DEVNULL)
        self.assertEqual(calls[0][1]["stdout"], asyncio.subprocess.PIPE)
        self.assertEqual(calls[0][1]["stderr"], asyncio.subprocess.PIPE)
        self.assertEqual(calls[0][1]["cwd"], "/tmp/code-bridge")
        self.assertEqual(events[0]["message"]["content"][0]["text"], "hello ")
        self.assertEqual(events[1]["message"]["content"][0]["text"], "gemini")
        self.assertEqual(events[0]["raw_event"]["type"], "content")
        self.assertEqual(events[0]["provider_id"], "google")
        self.assertEqual(events[2]["type"], "result")
        self.assertEqual(events[2]["result"], "hello gemini")
        self.assertEqual(events[2]["usage"], {"input_tokens": 3})
        self.assertEqual(events[2]["raw_event"]["type"], "completed")

    async def test_tool_result_normalizes_to_assistant_tool_result_block(self):
        session = self._session()

        await session._handle_jsonl_line(
            _jsonl({"type": "tool_result", "tool_use_id": "tool-1", "result": "done"})
        )

        event = await session._event_queue.get()
        self.assertEqual(event["type"], "assistant")
        block = event["message"]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "tool-1")
        self.assertEqual(block["content"], "done")
        self.assertEqual(event["raw_event"]["type"], "tool_result")

    async def test_abort_current_turn_sends_sigint(self):
        session = self._session()
        process = _FakeProcess(returncode=None)
        session._process = process
        session._turn_in_progress = True

        aborted = await session.abort_current_turn()

        self.assertTrue(aborted)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertFalse(session._turn_in_progress)


if __name__ == "__main__":
    unittest.main()
