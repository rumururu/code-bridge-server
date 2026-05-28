import os
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.history import ClaudeHistoryAdapter, GeminiHistoryAdapter
from llm.history.claude import _slugify_path


class _FakeSession:
    def __init__(self):
        self.resumed_session_id = None

    async def resume_session(self, session_id):
        self.resumed_session_id = session_id


class HistoryProvidersTest(unittest.IsolatedAsyncioTestCase):
    def _write_claude_session(self, home: Path, project_path: str, session_id: str) -> None:
        directory = home / ".claude" / "projects" / _slugify_path(project_path)
        directory.mkdir(parents=True)
        (directory / f"{session_id}.jsonl").write_text(
            "\n".join(
                [
                    (
                        '{"type":"user","timestamp":"2026-05-05T01:00:00Z",'
                        '"message":{"content":[{"type":"text","text":"hello claude"}]}}'
                    ),
                    (
                        '{"type":"assistant","timestamp":"2026-05-05T01:00:01Z",'
                        '"message":{"content":"hello user"}}'
                    ),
                ]
            ),
            encoding="utf-8",
        )

    async def test_claude_adapter_lists_messages_and_resumes_project_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project_path = "/tmp/demo-project"
            session_id = "claude-session"
            self._write_claude_session(home, project_path, session_id)
            adapter = ClaudeHistoryAdapter()
            fake_session = _FakeSession()

            with patch.dict(os.environ, {"HOME": tmp}):
                listed = adapter.list_sessions(project_path)
                messages = adapter.get_messages(project_path, session_id)
                resumed = await adapter.resume_session(project_path, session_id, fake_session)

        self.assertEqual(listed.provider_id, "anthropic")
        self.assertEqual(listed.sessions[0].session_id, session_id)
        self.assertEqual(listed.sessions[0].preview, "hello claude")
        self.assertEqual(listed.sessions[0].scope, "project")
        self.assertTrue(listed.sessions[0].resumable)
        self.assertEqual(
            messages.messages,
            [
                {
                    "type": "user",
                    "content": "hello claude",
                    "timestamp": "2026-05-05T01:00:00Z",
                },
                {
                    "type": "assistant",
                    "content": "hello user",
                    "timestamp": "2026-05-05T01:00:01Z",
                },
            ],
        )
        self.assertTrue(resumed.ok)
        self.assertEqual(fake_session.resumed_session_id, session_id)

    async def test_claude_adapter_matches_cli_slug_for_underscore_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project_path = "/Volumes/home_extension/AndroidStudioProjects/lottosignal"
            session_id = "claude-session"
            self._write_claude_session(home, project_path, session_id)
            adapter = ClaudeHistoryAdapter()

            with patch.dict(os.environ, {"HOME": tmp}):
                listed = adapter.list_sessions(project_path)

        self.assertEqual(
            _slugify_path(project_path),
            "-Volumes-home-extension-AndroidStudioProjects-lottosignal",
        )
        self.assertEqual(listed.sessions[0].session_id, session_id)

    async def test_claude_adapter_restores_cli_events_as_system_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project_path = "/tmp/demo-project"
            session_id = "claude-events"
            directory = home / ".claude" / "projects" / _slugify_path(project_path)
            directory.mkdir(parents=True)
            (directory / f"{session_id}.jsonl").write_text(
                "\n".join(
                    json.dumps(item)
                    for item in [
                        {
                            "type": "permission-mode",
                            "permissionMode": "default",
                            "timestamp": "2026-05-05T01:00:00Z",
                        },
                        {
                            "type": "user",
                            "timestamp": "2026-05-05T01:00:01Z",
                            "message": {"content": "run tests"},
                        },
                        {
                            "type": "assistant",
                            "timestamp": "2026-05-05T01:00:02Z",
                            "message": {
                                "content": [
                                    {"type": "thinking", "thinking": "Need pytest."},
                                    {
                                        "type": "tool_use",
                                        "id": "tool-1",
                                        "name": "Bash",
                                        "input": {"cmd": "pytest"},
                                    },
                                    {"type": "text", "text": "Running tests."},
                                ]
                            },
                        },
                        {
                            "type": "user",
                            "timestamp": "2026-05-05T01:00:03Z",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "tool-1",
                                        "content": "2 passed",
                                    }
                                ]
                            },
                        },
                        {
                            "type": "attachment",
                            "timestamp": "2026-05-05T01:00:04Z",
                            "attachment": {
                                "type": "file-history-snapshot",
                                "snapshot": {"trackedFileBackups": {"a.py": {}}},
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            adapter = ClaudeHistoryAdapter()

            with patch.dict(os.environ, {"HOME": tmp}):
                messages = adapter.get_messages(project_path, session_id).messages

        self.assertEqual([item["type"] for item in messages], [
            "system",
            "user",
            "assistant",
            "system",
            "system",
            "system",
            "system",
        ])
        contents = "\n".join(item["content"] for item in messages)
        self.assertIn("[permission mode] default", contents)
        self.assertIn("[thinking]", contents)
        self.assertIn("[tool call] Bash (tool-1)", contents)
        self.assertIn("[tool result] tool-1", contents)
        self.assertIn("[file history snapshot] tracked files: 1", contents)

    async def test_gemini_adapter_returns_typed_unsupported_results(self):
        adapter = GeminiHistoryAdapter()

        listed = adapter.list_sessions("/tmp/demo-project")
        messages = adapter.get_messages("/tmp/demo-project", "gemini-session")
        resumed = await adapter.resume_session("/tmp/demo-project", "gemini-session", None)

        self.assertEqual(listed.sessions, [])
        self.assertFalse(listed.resumable)
        self.assertIsInstance(listed.unsupported_reason, str)
        self.assertEqual(messages.messages, [])
        self.assertFalse(messages.resumable)
        self.assertFalse(resumed.ok)
        self.assertFalse(resumed.resumable)
        self.assertIsInstance(resumed.unsupported_reason, str)


if __name__ == "__main__":
    unittest.main()
