import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from chat.chat_session_service import ChatProviderSelection
from routes import chat_sessions


class _ProjectManager:
    def get_project(self, project_name):
        if project_name == "demo":
            return {"path": "/tmp/demo-project"}
        return None


class _FakeSession:
    def __init__(self):
        self.resumed_session_id = None

    async def resume_session(self, session_id):
        self.resumed_session_id = session_id


class ChatSessionsRoutesTest(unittest.IsolatedAsyncioTestCase):
    def _write_codex_fixture(
        self,
        root: Path,
        *,
        session_id: str = "019df000-1111-7222-8333-abcdefabcdef",
        cwd: str = "/tmp/demo-project",
        thread_name: str = "Codex prior task",
    ) -> str:
        (root / "sessions" / "2026" / "05" / "05").mkdir(parents=True)
        (root / "session_index.jsonl").write_text(
            (
                '{"id":"%s","thread_name":"%s",'
                '"updated_at":"2026-05-05T01:02:03Z"}\n'
            ) % (session_id, thread_name),
            encoding="utf-8",
        )
        rollout = root / "sessions" / "2026" / "05" / "05" / (
            f"rollout-2026-05-05T10-02-03-{session_id}.jsonl"
        )
        rollout.write_text(
            "\n".join(
                [
                    (
                        '{"timestamp":"2026-05-05T01:02:03Z","type":"session_meta",'
                        f'"payload":{{"cwd":"{cwd}"}}}}'
                    ),
                    (
                        '{"timestamp":"2026-05-05T01:02:04Z","type":"event_msg",'
                        '"payload":{"type":"user_message","message":"hello codex"}}'
                    ),
                    (
                        '{"timestamp":"2026-05-05T01:02:05Z","type":"event_msg",'
                        '"payload":{"type":"agent_message","message":"hello user"}}'
                    ),
                    (
                        '{"timestamp":"2026-05-05T01:02:06Z","type":"response_item",'
                        '"payload":{"type":"function_call","name":"exec_command",'
                        '"call_id":"call-1","arguments":"{\\"cmd\\":\\"pytest\\"}"}}'
                    ),
                    (
                        '{"timestamp":"2026-05-05T01:02:07Z","type":"response_item",'
                        '"payload":{"type":"function_call_output","call_id":"call-1",'
                        '"output":"2 passed"}}'
                    ),
                ]
            ),
            encoding="utf-8",
        )
        return session_id

    async def test_openai_list_sessions_reads_codex_session_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = self._write_codex_fixture(root)
            with patch.dict(os.environ, {"CODEX_HOME": tmp}), patch.object(
                chat_sessions, "get_project_manager", return_value=_ProjectManager()
            ), patch.object(
                chat_sessions,
                "get_chat_provider_selection",
                return_value=ChatProviderSelection("openai", "OpenAI", "gpt-5"),
            ):
                result = await chat_sessions.list_sessions("demo")

        self.assertEqual(result["sessions"][0]["session_id"], session_id)
        self.assertEqual(result["sessions"][0]["preview"], "Codex prior task")
        self.assertGreater(result["sessions"][0]["updated_at"], 0)
        self.assertGreater(result["sessions"][0]["size_bytes"], 0)
        self.assertEqual(result["sessions"][0]["provider_id"], "openai")
        self.assertEqual(result["sessions"][0]["scope"], "project")
        self.assertEqual(result["sessions"][0]["cwd"], "/tmp/demo-project")
        self.assertTrue(result["sessions"][0]["resumable"])
        self.assertEqual(result["provider_id"], "openai")

    async def test_openai_list_sessions_filters_unrelated_codex_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matching_id = self._write_codex_fixture(
                root,
                session_id="019df000-1111-7222-8333-aaaaaaaaaaaa",
                cwd="/tmp/demo-project",
                thread_name="matching",
            )
            with (root / "session_index.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(
                    '{"id":"019df000-1111-7222-8333-bbbbbbbbbbbb",'
                    '"thread_name":"unrelated","updated_at":"2026-05-05T01:02:04Z"}\n'
                )
            unrelated = root / "sessions" / "2026" / "05" / "05" / (
                "rollout-2026-05-05T10-02-04-019df000-1111-7222-8333-bbbbbbbbbbbb.jsonl"
            )
            unrelated.write_text(
                '{"timestamp":"2026-05-05T01:02:04Z","type":"session_meta",'
                '"payload":{"cwd":"/tmp/other-project"}}\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": tmp}), patch.object(
                chat_sessions, "get_project_manager", return_value=_ProjectManager()
            ), patch.object(
                chat_sessions,
                "get_chat_provider_selection",
                return_value=ChatProviderSelection("openai", "OpenAI", "gpt-5"),
            ):
                result = await chat_sessions.list_sessions("demo")

        self.assertEqual([session["session_id"] for session in result["sessions"]], [matching_id])

    async def test_openai_session_messages_read_codex_rollout_event_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = self._write_codex_fixture(root)
            with patch.dict(os.environ, {"CODEX_HOME": tmp}), patch.object(
                chat_sessions, "get_project_manager", return_value=_ProjectManager()
            ), patch.object(
                chat_sessions,
                "get_chat_provider_selection",
                return_value=ChatProviderSelection("openai", "OpenAI", "gpt-5"),
            ):
                result = await chat_sessions.get_session_messages("demo", session_id)

        self.assertEqual(result["session_id"], session_id)
        self.assertEqual(result["provider_id"], "openai")
        self.assertEqual(
            result["messages"],
            [
                {
                    "type": "user",
                    "content": "hello codex",
                    "timestamp": "2026-05-05T01:02:04Z",
                },
                {
                    "type": "assistant",
                    "content": "hello user",
                    "timestamp": "2026-05-05T01:02:05Z",
                },
                {
                    "type": "system",
                    "content": '[tool call] exec_command (call-1)\n{"cmd":"pytest"}',
                    "timestamp": "2026-05-05T01:02:06Z",
                },
                {
                    "type": "system",
                    "content": "[tool result] call-1\n2 passed",
                    "timestamp": "2026-05-05T01:02:07Z",
                },
            ],
        )

    async def test_openai_session_messages_reject_unrelated_codex_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = self._write_codex_fixture(root, cwd="/tmp/other-project")
            with patch.dict(os.environ, {"CODEX_HOME": tmp}), patch.object(
                chat_sessions, "get_project_manager", return_value=_ProjectManager()
            ), patch.object(
                chat_sessions,
                "get_chat_provider_selection",
                return_value=ChatProviderSelection("openai", "OpenAI", "gpt-5"),
            ):
                with self.assertRaises(HTTPException) as ctx:
                    await chat_sessions.get_session_messages("demo", session_id)

        self.assertEqual(ctx.exception.status_code, 403)

    async def test_openai_resume_uses_selected_provider_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = self._write_codex_fixture(root)
            fake_session = _FakeSession()
            calls = []

            async def fake_create_chat_session(project_name, project_path, selection):
                calls.append((project_name, project_path, selection))
                return fake_session

            selection = ChatProviderSelection("openai", "OpenAI", "gpt-5")
            with patch.dict(os.environ, {"CODEX_HOME": tmp}), patch.object(
                chat_sessions, "get_project_manager", return_value=_ProjectManager()
            ), patch.object(
                chat_sessions,
                "get_chat_provider_selection",
                return_value=selection,
            ), patch.object(
                chat_sessions,
                "create_chat_session",
                fake_create_chat_session,
            ):
                result = await chat_sessions.resume_session("demo", session_id)

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["session_id"], session_id)
        self.assertEqual(result["provider_id"], "openai")
        self.assertTrue(result["resumable"])
        self.assertEqual(fake_session.resumed_session_id, session_id)
        self.assertEqual(calls, [("demo", "/tmp/demo-project", selection)])

    async def test_openai_resume_rejects_unrelated_codex_cwd_before_session_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = self._write_codex_fixture(root, cwd="/tmp/other-project")
            calls = []

            async def fake_create_chat_session(project_name, project_path, selection):
                calls.append((project_name, project_path, selection))
                return _FakeSession()

            with patch.dict(os.environ, {"CODEX_HOME": tmp}), patch.object(
                chat_sessions, "get_project_manager", return_value=_ProjectManager()
            ), patch.object(
                chat_sessions,
                "get_chat_provider_selection",
                return_value=ChatProviderSelection("openai", "OpenAI", "gpt-5"),
            ), patch.object(
                chat_sessions,
                "create_chat_session",
                fake_create_chat_session,
            ):
                with self.assertRaises(HTTPException) as ctx:
                    await chat_sessions.resume_session("demo", session_id)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(calls, [])

    async def test_openai_resume_rejects_unknown_codex_cwd_before_session_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = "019df000-1111-7222-8333-cccccccccccc"
            (root / "sessions" / "2026" / "05" / "05").mkdir(parents=True)
            (root / "session_index.jsonl").write_text(
                (
                    '{"id":"%s","thread_name":"unknown cwd",'
                    '"updated_at":"2026-05-05T01:02:03Z"}\n'
                )
                % session_id,
                encoding="utf-8",
            )
            (root / "sessions" / "2026" / "05" / "05" / f"rollout-{session_id}.jsonl").write_text(
                '{"timestamp":"2026-05-05T01:02:04Z","type":"event_msg",'
                '"payload":{"type":"user_message","message":"hello codex"}}\n',
                encoding="utf-8",
            )
            calls = []

            async def fake_create_chat_session(project_name, project_path, selection):
                calls.append((project_name, project_path, selection))
                return _FakeSession()

            with patch.dict(os.environ, {"CODEX_HOME": tmp}), patch.object(
                chat_sessions, "get_project_manager", return_value=_ProjectManager()
            ), patch.object(
                chat_sessions,
                "get_chat_provider_selection",
                return_value=ChatProviderSelection("openai", "OpenAI", "gpt-5"),
            ), patch.object(
                chat_sessions,
                "create_chat_session",
                fake_create_chat_session,
            ):
                listed = await chat_sessions.list_sessions("demo")
                with self.assertRaises(HTTPException) as ctx:
                    await chat_sessions.resume_session("demo", session_id)

        self.assertEqual(listed["sessions"], [])
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(calls, [])

    async def test_google_list_sessions_does_not_return_claude_sessions(self):
        with patch.object(
            chat_sessions, "get_project_manager", return_value=_ProjectManager()
        ), patch.object(
            chat_sessions,
            "get_chat_provider_selection",
            return_value=ChatProviderSelection("google", "Gemini", "gemini-2.5-pro"),
        ):
            result = await chat_sessions.list_sessions("demo")

        self.assertEqual(result["sessions"], [])
        self.assertEqual(result["provider_id"], "google")
        self.assertFalse(result["resumable"])
        self.assertIsInstance(result["unsupported_reason"], str)


if __name__ == "__main__":
    unittest.main()
