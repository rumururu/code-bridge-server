import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store  # noqa: E402
from core import database  # noqa: E402
from routes import agent_browser_stream  # noqa: E402


class _FakeBrowserStreamSession:
    def __init__(self, browser_session, *, viewport_width=1280, viewport_height=720):
        self.browser_session = browser_session
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.closed = False

    async def start(self):
        return None

    async def snapshot(self, *, image_type="jpeg", quality=70):
        return {
            "mime_type": "image/jpeg",
            "encoding": "base64",
            "data": "ZmFrZQ==",
            "width": self.viewport_width,
            "height": self.viewport_height,
            "url": "https://example.test",
            "title": "Example",
        }

    async def handle_client_message(self, message):
        return {
            "type": "browser_stream.input_result",
            "input_type": message.get("type"),
            "ok": True,
            "url": "https://example.test",
            "title": "Example",
        }

    async def close(self):
        self.closed = True


class AgentBrowserStreamRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "browser_stream_routes.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None

        app = FastAPI()
        app.include_router(agent_browser_stream.router)
        self.client = TestClient(app)
        self.store = agent_store.get_agent_store()

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _browser_handoff_fixture(self):
        agent = self.store.create_agent(name="stream bot", provider_id="openai")
        task = self.store.create_task(
            title="Browser stream",
            assigned_agent_id=agent["id"],
            goal="Complete browser handoff",
        )
        run = self.store.create_run(
            task_id=task["id"],
            agent_id=agent["id"],
            provider_id="openai",
            title="Browser stream run",
        )
        self.store.update_task(task["id"], {"run_id": run["id"], "status": "waiting_for_user"})
        step = self.store.create_task_step(
            task_id=task["id"],
            run_id=run["id"],
            title="Browser gate",
            status="waiting_for_user",
            input={
                "workflow_step_id": "browser_gate",
                "workflow_type": "browser_action",
            },
            output={},
        )
        session = browser_session_store.get_browser_session_store().create(
            run_id=run["id"],
            task_id=task["id"],
            step_id=step["id"],
            workflow_step_id="browser_gate",
            status="waiting_for_user",
            current_url="https://example.test",
        )
        self.store.update_task_step(
            step["id"],
            {
                "output": {
                    "browser_session_id": session["id"],
                    "checkpoint": {
                        "status": "waiting_for_user",
                        "reason": "browser_blocked",
                        "prompt": "Finish in browser.",
                        "workflow_step_id": "browser_gate",
                        "browser_session_id": session["id"],
                        "resume": "same_step",
                        "required_user_action": "Use streamed browser.",
                    },
                }
            },
        )
        return task, run, session

    def test_browser_handoff_stream_sends_ready_frame_and_accepts_input(self):
        task, _run, session = self._browser_handoff_fixture()

        with patch(
            "routes.agent_browser_stream.validate_api_key_for_current_server",
            return_value=SimpleNamespace(success=True, api_key="test", error=None),
        ), patch(
            "routes.agent_browser_stream.PlaywrightBrowserStreamSession",
            _FakeBrowserStreamSession,
        ):
            with self.client.websocket_connect(
                f"/ws/agent/tasks/{task['id']}/browser-handoff/stream?fps=1"
            ) as ws:
                ready = ws.receive_json()
                frame = ws.receive_json()
                ws.send_json({"type": "tap", "x": 0.5, "y": 0.5, "normalized": True})
                result = _receive_until(ws, "browser_stream.input_result")
                ws.send_json(
                    {
                        "type": "type_text",
                        "text": "secret-password",
                        "sensitive": True,
                    }
                )
                sensitive_result = _receive_until(ws, "browser_stream.input_result")
                ws.send_json({"type": "close"})
                closed = _receive_until(ws, "browser_stream.closed")

        self.assertEqual(ready["type"], "browser_stream.ready")
        self.assertEqual(ready["browser_session"]["id"], session["id"])
        self.assertEqual(frame["type"], "browser_stream.frame")
        self.assertEqual(frame["data"], "ZmFrZQ==")
        self.assertEqual(result["input_type"], "tap")
        self.assertTrue(result["ok"])
        self.assertEqual(sensitive_result["input_type"], "type_text")
        self.assertTrue(sensitive_result["ok"])
        self.assertTrue(sensitive_result["sensitive"])
        self.assertNotIn("secret-password", str(sensitive_result))
        self.assertEqual(closed["type"], "browser_stream.closed")

    def test_browser_handoff_stream_reports_missing_handoff(self):
        task = self.store.create_task(title="No handoff", goal="No checkpoint")

        with patch(
            "routes.agent_browser_stream.validate_api_key_for_current_server",
            return_value=SimpleNamespace(success=True, api_key="test", error=None),
        ):
            with self.client.websocket_connect(
                f"/ws/agent/tasks/{task['id']}/browser-handoff/stream"
            ) as ws:
                error = ws.receive_json()

        self.assertEqual(error["type"], "browser_stream.error")
        self.assertEqual(error["code"], "no_active_browser_handoff")

    def test_browser_handoff_stream_restores_missing_live_session(self):
        task, _run, session = self._browser_handoff_fixture()

        class _FakeRuntime:
            def __init__(self, browser_session):
                self.browser_session = browser_session

            async def set_viewport(self, width, height):
                self.browser_session = {
                    **self.browser_session,
                    "viewport_width": width,
                    "viewport_height": height,
                }

            async def sync_state(self, *, save_storage):
                return None

            async def snapshot(self, *, image_type="jpeg", quality=70):
                return {
                    "mime_type": "image/jpeg",
                    "encoding": "base64",
                    "data": "cmVzdG9yZWQ=",
                    "width": 1280,
                    "height": 720,
                    "url": self.browser_session["current_url"],
                    "title": self.browser_session["title"] or "Restored",
                }

            async def handle_client_message(self, message):
                return {"type": "browser_stream.input_result", "ok": True}

        class _FakeRuntimeManager:
            def get(self, session_id):
                return None

            async def open_session(
                self,
                browser_session,
                *,
                input_storage_state_path=None,
                viewport_width=1280,
                viewport_height=720,
            ):
                runtime = _FakeRuntime(browser_session)
                await runtime.set_viewport(viewport_width, viewport_height)
                return runtime

        with patch(
            "routes.agent_browser_stream.validate_api_key_for_current_server",
            return_value=SimpleNamespace(success=True, api_key="test", error=None),
        ), patch(
            "agent.browser_stream_session.get_browser_runtime_manager",
            return_value=_FakeRuntimeManager(),
        ):
            with self.client.websocket_connect(
                f"/ws/agent/tasks/{task['id']}/browser-handoff/stream?fps=1"
            ) as ws:
                ready = ws.receive_json()
                frame = ws.receive_json()
                ws.send_json({"type": "close"})

        self.assertEqual(ready["type"], "browser_stream.ready")
        self.assertEqual(ready["browser_session"]["id"], session["id"])
        self.assertEqual(frame["type"], "browser_stream.frame")
        self.assertEqual(frame["url"], "https://example.test")


def _receive_until(ws, message_type):
    for _ in range(5):
        payload = ws.receive_json()
        if payload.get("type") == message_type:
            return payload
    raise AssertionError(f"Did not receive {message_type}")


if __name__ == "__main__":
    unittest.main()
