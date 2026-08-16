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
from chat.chat_stream_service import (
    TurnState,
    _approval_display,
    _approval_target_details,
    _handle_control_request,
)
from core import database
from policy import policy_store
from policy.policy_engine import decide_policy


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


class FileReadPolicyEscalationTest(unittest.TestCase):
    """`file.read` is allowed *by default*, never allowed *always*.

    Mapping the read-only tools to an `ALLOW_OPERATIONS` member is only safe
    because `decide_policy` runs the path guard and secret classifier over the
    request's own target and escalates away from `allow`. If any assertion
    below flips to `allow`, B4 is not safe to ship: an unattended agent could
    read a private key without anyone being asked.
    """

    WORKSPACE = "/tmp/cb-policy-workspace"

    def _effect(self, path: str, *, workspace_root: str | None = WORKSPACE) -> str:
        details = {"path": path}
        if workspace_root:
            details["workspace_root"] = workspace_root
        return decide_policy("file.read", details=details)["effect"]

    def test_an_ordinary_source_file_inside_the_workspace_is_allowed(self):
        decision = decide_policy(
            "file.read",
            details={"path": f"{self.WORKSPACE}/lib/main.dart", "workspace_root": self.WORKSPACE},
        )
        self.assertEqual(decision["effect"], "allow")
        self.assertFalse(decision["approval_required"])

    def test_a_private_key_is_not_auto_approved(self):
        for path in ("~/.ssh/id_rsa", str(Path.home() / ".ssh" / "id_ed25519")):
            with self.subTest(path=path):
                decision = decide_policy(
                    "file.read",
                    details={"path": path, "workspace_root": self.WORKSPACE},
                )
                self.assertEqual(decision["effect"], "desktop_only")
                self.assertTrue(decision["approval_required"])

    def test_a_dotenv_inside_the_workspace_is_not_auto_approved(self):
        # The case the path guard used to miss entirely: unlike an SSH key,
        # a dotenv lives *inside* the project, so the workspace-boundary
        # check reads it as internal and would have allowed it.
        for name in (".env", ".env.local", "assets/.env", ".envrc"):
            with self.subTest(name=name):
                self.assertEqual(self._effect(f"{self.WORKSPACE}/{name}"), "desktop_only")

    def test_repo_credentials_are_not_auto_approved(self):
        for name in (
            "server/server_info.json",
            "server/firebase_config.json",
            "server/firebase_service_account.json",
            "android/app/google-services.json",
            "ios/fastlane/AuthKey_ABC123.p8",
            "lib/secrets.dart",
        ):
            with self.subTest(name=name):
                self.assertEqual(self._effect(f"{self.WORKSPACE}/{name}"), "desktop_only")

    def test_reading_outside_the_workspace_still_asks(self):
        self.assertEqual(self._effect("/Users/someone/Documents/taxes.pdf"), "confirm_each")

    def test_a_run_without_a_registered_cwd_fails_closed(self):
        # No workspace root means the guard cannot tell inside from outside,
        # so it must ask rather than assume.
        self.assertEqual(
            self._effect(f"{self.WORKSPACE}/lib/main.dart", workspace_root=None),
            "confirm_each",
        )

    def test_system_paths_stay_forbidden(self):
        self.assertEqual(self._effect("/etc/passwd"), "forbidden")


class ApprovalTargetPromotionTest(unittest.TestCase):
    """The tool's target has to reach the keys the classifiers actually read.

    `decide_policy` looks at `details["path"]` / `details["paths"]`. The tool's
    own target lives under `details["input"]["file_path"]`, where nothing
    looks. Without this promotion the `file.read` allow would be
    unconditional.
    """

    WORKSPACE = "/tmp/cb-target-workspace"

    def test_read_promotes_its_file_path(self):
        promoted = _approval_target_details(
            "Read",
            {"file_path": f"{self.WORKSPACE}/lib/main.dart"},
            workspace_root=self.WORKSPACE,
        )
        self.assertEqual(promoted["path"], f"{self.WORKSPACE}/lib/main.dart")
        self.assertEqual(promoted["paths"], [f"{self.WORKSPACE}/lib/main.dart"])

    def test_a_relative_target_resolves_against_the_workspace_not_the_server_cwd(self):
        promoted = _approval_target_details(
            "Read", {"file_path": "lib/main.dart"}, workspace_root=self.WORKSPACE
        )
        self.assertEqual(promoted["path"], f"{self.WORKSPACE}/lib/main.dart")

    def test_a_tilde_target_is_left_for_the_path_guard_to_expand(self):
        promoted = _approval_target_details(
            "Read", {"file_path": "~/.ssh/id_rsa"}, workspace_root=self.WORKSPACE
        )
        self.assertEqual(promoted["path"], "~/.ssh/id_rsa")

    def test_glob_promotes_its_pattern_because_a_glob_pattern_is_a_path(self):
        promoted = _approval_target_details(
            "Glob", {"pattern": "~/.ssh/*"}, workspace_root=self.WORKSPACE
        )
        self.assertIn("~/.ssh/*", promoted["paths"])

    def test_grep_does_not_promote_its_pattern_because_that_one_is_a_regex(self):
        promoted = _approval_target_details(
            "Grep",
            {"pattern": r"api[_-]?key\s*=", "path": f"{self.WORKSPACE}/server"},
            workspace_root=self.WORKSPACE,
        )
        self.assertEqual(promoted["paths"], [f"{self.WORKSPACE}/server"])

    def test_an_unknown_tool_only_trusts_keys_that_say_they_are_paths(self):
        promoted = _approval_target_details(
            "linear__create_issue",
            {"query": "login is broken", "path": f"{self.WORKSPACE}/notes.md"},
            workspace_root=self.WORKSPACE,
        )
        self.assertEqual(promoted["paths"], [f"{self.WORKSPACE}/notes.md"])

    def test_no_recognisable_target_promotes_nothing(self):
        self.assertEqual(
            _approval_target_details("WebSearch", {"query": "flutter"}, workspace_root=None),
            {},
        )


class ApprovalDisplayContractTest(unittest.TestCase):
    """`display` is the app's whole reason to stop dumping raw JSON.

    Keys are stable identifiers, not sentences: the app ships four locales
    and renders its own wording from `action`.
    """

    def test_read_and_write_and_run_normalize(self):
        for tool_name, tool_input, expected in (
            ("Read", {"file_path": "/w/a.dart"}, {"action": "read_file", "target": "/w/a.dart"}),
            (
                "NotebookRead",
                {"notebook_path": "/w/a.ipynb"},
                {"action": "read_file", "target": "/w/a.ipynb"},
            ),
            ("Bash", {"command": "npm test"}, {"action": "run_command", "target": "npm test"}),
            ("Edit", {"file_path": "/w/b.dart"}, {"action": "write_file", "target": "/w/b.dart"}),
            ("Write", {"file_path": "/w/c.dart"}, {"action": "write_file", "target": "/w/c.dart"}),
            (
                "MultiEdit",
                {"file_path": "/w/d.dart"},
                {"action": "write_file", "target": "/w/d.dart"},
            ),
        ):
            with self.subTest(tool_name=tool_name):
                self.assertEqual(_approval_display(tool_name, tool_input), expected)

    def test_an_unknown_tool_keeps_its_own_name_verbatim(self):
        self.assertEqual(
            _approval_display("linear__create_issue", {"query": "login is broken"}),
            {"action": "linear__create_issue", "target": "login is broken"},
        )

    def test_the_server_never_writes_a_sentence(self):
        # A localized sentence built here would be wrong in three of the
        # app's four locales. `action` must stay an identifier.
        for tool_name in ("Read", "Bash", "Edit", "Glob", "some_mcp_tool"):
            with self.subTest(tool_name=tool_name):
                action = _approval_display(tool_name, {})["action"]
                self.assertNotIn(" ", action)

    def test_a_missing_target_is_none_rather_than_a_guess(self):
        self.assertIsNone(_approval_display("Read", {})["target"])
        self.assertIsNone(_approval_display("Bash", {"description": "run tests"})["target"])


class ReadPermissionEndToEndTest(unittest.IsolatedAsyncioTestCase):
    """The whole B4 claim, exercised through the real control-request path."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_read_policy_test.db"
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        self.workspace = Path(self._tmp.name) / "workspace"
        self.workspace.mkdir()
        self.run = agent_store.get_agent_store().create_run(
            project_name="demo",
            provider_id="claude",
            model="sonnet",
            title="Read a file",
            goal="Summarize the build log",
            cwd=str(self.workspace),
        )

    def tearDown(self):
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    async def _read(self, file_path: str, *, tool_name: str = "Read"):
        websocket = _FakeWebSocket(agent_run_id=self.run["id"])
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
                        "tool_name": tool_name,
                        "tool_use_id": "tool-1",
                        "input": {"file_path": file_path},
                    },
                },
                "demo",
            )
        return result, websocket, retry_stream

    async def test_reading_a_project_file_no_longer_stops_the_run(self):
        target = self.workspace / "lib" / "main.dart"
        result, websocket, retry_stream = await self._read(str(target))

        self.assertTrue(result)
        retry_stream.assert_awaited_once()
        self.assertEqual(retry_stream.await_args.kwargs["retry_from_permission"], True)
        self.assertNotIn("permission_required", [item.get("type") for item in websocket.sent])
        self.assertIn(
            "permission.auto_approved",
            [item.get("event") for item in websocket.sent if item.get("type") == "app_event"],
        )
        # No standing rule was involved — this is the built-in default.
        self.assertEqual(policy_store.get_policy_rule_store().list_rules(), [])
        self.assertEqual(approval_store.get_approval_store().list_pending(), [])

    async def test_reading_a_private_key_still_stops_the_run(self):
        result, websocket, retry_stream = await self._read("~/.ssh/id_rsa")

        self.assertFalse(result)
        retry_stream.assert_not_awaited()
        permission_event = websocket.sent[0]
        self.assertEqual(permission_event["type"], "permission_required")
        self.assertTrue(permission_event["desktop_only"])
        pending = approval_store.get_approval_store().list_pending(run_id=self.run["id"])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["operation"], "file.read")

    async def test_reading_a_dotenv_inside_the_project_still_stops_the_run(self):
        result, websocket, _ = await self._read(str(self.workspace / ".env"))

        self.assertFalse(result)
        self.assertEqual(websocket.sent[0]["type"], "permission_required")
        self.assertTrue(websocket.sent[0]["desktop_only"])

    async def test_a_standing_allow_on_terminal_does_not_leak_into_reads(self):
        # The rule that caused the inversion in the first place. It must keep
        # doing exactly what it says and nothing more.
        policy_store.get_policy_rule_store().create_rule(
            scope="global", operation="process.terminal", effect="allow"
        )
        result, websocket, _ = await self._read("~/.ssh/id_rsa")
        self.assertFalse(result)
        self.assertTrue(websocket.sent[0]["desktop_only"])

    async def test_a_standing_allow_on_file_read_cannot_override_the_escalation(self):
        # `decide_policy_with_rules` returns the built-in decision untouched
        # once it is `desktop_only` or `forbidden`, so a blanket "allow all
        # reads" rule — the thing a frustrated user is most likely to create
        # — still cannot hand an unattended agent a private key.
        policy_store.get_policy_rule_store().create_rule(
            scope="global", operation="file.read", effect="allow"
        )
        result, websocket, retry_stream = await self._read("~/.ssh/id_rsa")
        self.assertFalse(result)
        retry_stream.assert_not_awaited()
        self.assertTrue(websocket.sent[0]["desktop_only"])
        # ...while the same rule leaves an ordinary project read allowed.
        ok, _, ok_retry = await self._read(str(self.workspace / "lib" / "main.dart"))
        self.assertTrue(ok)
        ok_retry.assert_awaited_once()

    async def test_the_approval_row_carries_the_display_contract_and_the_path(self):
        target = self.workspace / "config" / "settings.pem"  # forces a pending row
        await self._read(str(target))

        pending = approval_store.get_approval_store().list_pending(run_id=self.run["id"])
        self.assertEqual(len(pending), 1)
        details = pending[0]["details"]
        self.assertEqual(details["display"], {"action": "read_file", "target": str(target)})
        self.assertEqual(details["path"], str(target))
        self.assertEqual(details["workspace_root"], str(self.workspace))

    async def test_the_permission_required_event_carries_display_too(self):
        # The interactive chat path has no approval row to read `display`
        # off, so the websocket denial has to carry it as well.
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
        self.assertEqual(
            websocket.sent[0]["denials"][0]["display"],
            {"action": "run_command", "target": "npm test"},
        )


if __name__ == "__main__":
    unittest.main()
