import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from approvals import approval_store
from audit import audit_store
from chat.chat_stream_service import TurnState, _handle_control_request
from core import database
from policy import policy_store


class _FakeWebSocket:
    def __init__(self, *, agent_run_id: str | None = None):
        self.sent = []
        if agent_run_id is not None:
            self.agent_run_id = agent_run_id

    async def send_json(self, data):
        self.sent.append(data)


class _FakeSession:
    provider_id = "claude"
    session_id = "native-1"

    async def approve_pending_permissions_and_retry(self):
        yield {"type": "result", "result": "allowed", "total_cost_usd": 0}

    async def deny_pending_permissions(self, message="Permission denied by user."):
        yield {"type": "result", "result": message, "total_cost_usd": 0}


class ChatPermissionApprovalTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_chat_approval_test.db"
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

    def tearDown(self):
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    async def test_tool_permission_prompt_creates_pending_approval_for_agent_run(self):
        run = agent_store.get_agent_store().create_run(
            project_name="demo",
            provider_id="claude",
            model="sonnet",
            title="Run command",
            goal="Execute tests",
            cwd="/tmp/demo",
        )
        websocket = _FakeWebSocket(agent_run_id=run["id"])
        state = TurnState(provider_id="claude", provider="claude", session_id="native-1")

        result = await _handle_control_request(
            websocket,
            _FakeSession(),
            state,
            {
                "request_id": "provider-req-1",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Bash",
                    "tool_use_id": "tool-1",
                    "input": {"command": "npm test", "api_key": "secret-value"},
                },
            },
            "demo",
        )

        self.assertFalse(result)
        permission_event = websocket.sent[0]
        self.assertEqual(permission_event["type"], "permission_required")
        self.assertTrue(permission_event["approval_id"].startswith("apr_"))
        self.assertEqual(
            permission_event["denials"][0]["approval_id"],
            permission_event["approval_id"],
        )
        self.assertFalse(permission_event["desktop_only"])

        pending = approval_store.get_approval_store().list_pending(run_id=run["id"])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["id"], permission_event["approval_id"])
        self.assertEqual(pending[0]["operation"], "process.terminal")
        self.assertEqual(pending[0]["details"]["input"]["command"], "npm test")

        audit_events = audit_store.get_audit_store().list_events(run_id=run["id"])
        self.assertEqual(audit_events[0]["decision"], "approval_requested")
        self.assertEqual(audit_events[0]["operation"], "process.terminal")
        self.assertEqual(
            audit_events[0]["payload"]["details"]["input"]["api_key"],
            "[redacted]",
        )

    async def test_tool_permission_prompt_without_agent_run_stays_event_only(self):
        websocket = _FakeWebSocket()
        state = TurnState(provider_id="claude", provider="claude", session_id="native-1")

        await _handle_control_request(
            websocket,
            _FakeSession(),
            state,
            {
                "request_id": "provider-req-1",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Bash",
                    "tool_use_id": "tool-1",
                    "input": {"command": "npm test"},
                },
            },
            "demo",
        )

        permission_event = websocket.sent[0]
        self.assertEqual(permission_event["type"], "permission_required")
        self.assertIsNone(permission_event["approval_id"])
        self.assertEqual(approval_store.get_approval_store().list_pending(), [])

    async def test_allow_policy_auto_approves_provider_permission(self):
        run = agent_store.get_agent_store().create_run(
            project_name="demo",
            provider_id="claude",
            model="sonnet",
            title="Run command",
            goal="Execute tests",
            cwd="/tmp/demo",
        )
        policy_store.get_policy_rule_store().create_rule(
            scope="global",
            operation="process.terminal",
            effect="allow",
        )
        websocket = _FakeWebSocket(agent_run_id=run["id"])
        state = TurnState(provider_id="claude", provider="claude", session_id="native-1")

        with patch(
            "chat.chat_stream_service.stream_claude_turn",
            new=AsyncMock(return_value=True),
        ) as retry_stream:
            result = await _handle_control_request(
                websocket,
                _FakeSession(),
                state,
                {
                    "request_id": "provider-req-1",
                    "request": {
                        "subtype": "can_use_tool",
                        "tool_name": "Bash",
                        "tool_use_id": "tool-1",
                        "input": {"command": "npm test"},
                    },
                },
                "demo",
            )

        self.assertTrue(result)
        retry_stream.assert_awaited_once()
        self.assertEqual(approval_store.get_approval_store().list_pending(), [])
        self.assertNotIn("permission_required", [item.get("type") for item in websocket.sent])
        self.assertIn(
            "permission.auto_approved",
            [item.get("event") for item in websocket.sent if item.get("type") == "app_event"],
        )

    async def test_desktop_only_permission_prompt_marks_remote_restriction(self):
        run = agent_store.get_agent_store().create_run(
            project_name="demo",
            provider_id="claude",
            model="sonnet",
            title="Run command",
            goal="Execute tests",
            cwd="/tmp/demo",
        )
        policy_store.get_policy_rule_store().create_rule(
            scope="global",
            operation="process.terminal",
            effect="desktop_only",
        )
        websocket = _FakeWebSocket(agent_run_id=run["id"])
        state = TurnState(provider_id="claude", provider="claude", session_id="native-1")

        result = await _handle_control_request(
            websocket,
            _FakeSession(),
            state,
            {
                "request_id": "provider-req-1",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Bash",
                    "tool_use_id": "tool-1",
                    "input": {"command": "npm test"},
                },
            },
            "demo",
        )

        self.assertFalse(result)
        permission_event = websocket.sent[0]
        self.assertTrue(permission_event["desktop_only"])
        self.assertTrue(permission_event["denials"][0]["desktop_only"])
        self.assertEqual(permission_event["policy"]["effect"], "desktop_only")


if __name__ == "__main__":
    unittest.main()
