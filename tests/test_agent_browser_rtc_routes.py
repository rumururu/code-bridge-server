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
from routes import agent_browser_rtc  # noqa: E402


class _FakeBrowserRtcPeerSession:
    def __init__(
        self,
        browser_session,
        *,
        viewport_width=1280,
        viewport_height=720,
        fps=12,
        quality=70,
        capture_mode="page",
        input_result_callback=None,
    ):
        self.browser_session = browser_session
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.fps = fps
        self.quality = quality
        self.capture_mode = capture_mode
        self.input_result_callback = input_result_callback
        self.candidates = []
        self.closed = False

    async def start(self):
        return None

    async def accept_offer(self, *, sdp, offer_type="offer"):
        return {"type": "answer", "sdp": f"answer-for-{offer_type}:{sdp}"}

    async def add_ice_candidate(self, candidate):
        self.candidates.append(candidate)

    async def close(self):
        self.closed = True


class AgentBrowserRtcRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "browser_rtc_routes.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None

        app = FastAPI()
        app.include_router(agent_browser_rtc.router)
        self.client = TestClient(app)
        self.store = agent_store.get_agent_store()

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _browser_handoff_fixture(self):
        agent = self.store.create_agent(name="rtc bot", provider_id="openai")
        task = self.store.create_task(
            title="Browser RTC",
            assigned_agent_id=agent["id"],
            goal="Complete browser handoff",
        )
        run = self.store.create_run(
            task_id=task["id"],
            agent_id=agent["id"],
            provider_id="openai",
            title="Browser RTC run",
        )
        self.store.update_task(
            task["id"],
            {"run_id": run["id"], "status": "waiting_for_user"},
        )
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

    def test_browser_handoff_rtc_signals_offer_answer(self):
        task, _run, session = self._browser_handoff_fixture()

        with patch(
            "routes.agent_browser_rtc.validate_api_key_for_current_server",
            return_value=SimpleNamespace(success=True, api_key="test", error=None),
        ), patch(
            "routes.agent_browser_rtc.BrowserRtcPeerSession",
            _FakeBrowserRtcPeerSession,
        ):
            with self.client.websocket_connect(
                f"/ws/agent/tasks/{task['id']}/browser-handoff/rtc?fps=12"
            ) as ws:
                ready = ws.receive_json()
                ws.send_json(
                    {
                        "type": "offer",
                        "sdp": "fake-offer",
                        "sdpType": "offer",
                    }
                )
                answer = ws.receive_json()
                ws.send_json(
                    {
                        "type": "candidate",
                        "candidate": {
                            "candidate": "candidate:1 1 UDP 1 127.0.0.1 9 typ host",
                            "sdpMid": "0",
                            "sdpMLineIndex": 0,
                        },
                    }
                )
                ws.send_json({"type": "close"})
                closed = ws.receive_json()

        self.assertEqual(ready["type"], "browser_rtc.ready")
        self.assertEqual(ready["browser_session"]["id"], session["id"])
        self.assertEqual(answer["type"], "answer")
        self.assertEqual(answer["sdp"], "answer-for-offer:fake-offer")
        self.assertEqual(closed["type"], "browser_rtc.closed")

    def test_browser_handoff_rtc_exposes_page_capture_mode(self):
        task, _run, _session = self._browser_handoff_fixture()
        created_sessions = []

        class CapturingFakeBrowserRtcPeerSession(_FakeBrowserRtcPeerSession):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_sessions.append(self)

        with patch(
            "routes.agent_browser_rtc.validate_api_key_for_current_server",
            return_value=SimpleNamespace(success=True, api_key="test", error=None),
        ), patch(
            "routes.agent_browser_rtc.BrowserRtcPeerSession",
            CapturingFakeBrowserRtcPeerSession,
        ):
            with self.client.websocket_connect(
                f"/ws/agent/tasks/{task['id']}/browser-handoff/rtc?capture=page"
            ) as ws:
                ready = ws.receive_json()
                ws.send_json({"type": "close"})
                ws.receive_json()

        self.assertEqual(ready["capture"], "page")
        self.assertEqual(created_sessions[0].capture_mode, "page")

    def test_browser_handoff_rtc_reports_missing_handoff(self):
        task = self.store.create_task(title="No handoff", goal="No checkpoint")

        with patch(
            "routes.agent_browser_rtc.validate_api_key_for_current_server",
            return_value=SimpleNamespace(success=True, api_key="test", error=None),
        ):
            with self.client.websocket_connect(
                f"/ws/agent/tasks/{task['id']}/browser-handoff/rtc"
            ) as ws:
                error = ws.receive_json()

        self.assertEqual(error["type"], "browser_rtc.error")
        self.assertEqual(error["code"], "no_active_browser_handoff")


if __name__ == "__main__":
    unittest.main()
