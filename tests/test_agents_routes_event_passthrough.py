"""Coverage for ``POST /api/agent/runs/{run_id}/event`` passthrough.

TASK_001 added ``call_id`` and ``parent_event_id`` columns to
``agent_events`` plus the matching fields on ``AgentEventCreate``. The
REST handler still needs to actually forward those body fields into
``store.append_event(...)`` (REVIEW_001 P2-1). These tests verify the
two new fields round-trip via the FastAPI handler and that existing
clients which omit them remain compatible.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from approvals import approval_store  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class AgentsRoutesEventPassthroughTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = (
            Path(self._tmp.name) / "code_bridge_event_passthrough.db"
        )
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

        # Create a run so /event has a valid target.
        run_response = self.client.post(
            "/api/agent/runs",
            json={
                "project_name": "demo",
                "provider_id": "anthropic",
                "model": "claude-sonnet-4-5",
                "title": "Event passthrough",
                "goal": "verify call_id forwarding",
                "cwd": "/tmp/demo",
            },
        )
        self.assertEqual(run_response.status_code, 200)
        self.run_id = run_response.json()["run"]["id"]

    def tearDown(self) -> None:
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _fetch_events(self) -> list[dict]:
        rows = self.client.get(f"/api/agent/runs/{self.run_id}/events").json()
        return rows.get("events", []) if isinstance(rows, dict) else rows

    # Case 1 — call_id flows through the route into the agent_events row.
    def test_call_id_in_body_is_persisted(self) -> None:
        response = self.client.post(
            f"/api/agent/runs/{self.run_id}/event",
            json={
                "event_type": "tool_use",
                "provider_id": "anthropic",
                "app_event": {"name": "Bash"},
                "call_id": "toolu_route_001",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["event"]["call_id"], "toolu_route_001")
        # And the GET endpoint surfaces the same value.
        events = self._fetch_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["call_id"], "toolu_route_001")
        self.assertIsNone(events[0]["parent_event_id"])

    # Case 2 — parent_event_id forwarded; pairs tool_result back to tool_use.
    def test_parent_event_id_in_body_is_persisted(self) -> None:
        use_response = self.client.post(
            f"/api/agent/runs/{self.run_id}/event",
            json={
                "event_type": "tool_use",
                "provider_id": "anthropic",
                "app_event": {"name": "Bash"},
                "call_id": "toolu_route_pair",
            },
        )
        self.assertEqual(use_response.status_code, 200)
        use_event_id = use_response.json()["event"]["id"]

        result_response = self.client.post(
            f"/api/agent/runs/{self.run_id}/event",
            json={
                "event_type": "tool_result",
                "provider_id": "anthropic",
                "app_event": {"output": "ok"},
                "call_id": "toolu_route_pair",
                "parent_event_id": use_event_id,
            },
        )
        self.assertEqual(result_response.status_code, 200)
        result_event = result_response.json()["event"]
        self.assertEqual(result_event["call_id"], "toolu_route_pair")
        self.assertEqual(result_event["parent_event_id"], use_event_id)

        # And both events round-trip via GET.
        events = self._fetch_events()
        self.assertEqual(len(events), 2)
        by_type = {row["event_type"]: row for row in events}
        self.assertEqual(by_type["tool_use"]["call_id"], "toolu_route_pair")
        self.assertEqual(by_type["tool_use"]["parent_event_id"], None)
        self.assertEqual(
            by_type["tool_result"]["call_id"], "toolu_route_pair"
        )
        self.assertEqual(
            by_type["tool_result"]["parent_event_id"], use_event_id
        )

    # Case 3 — backward compatibility: clients that omit both fields still work.
    def test_legacy_body_without_call_id_or_parent_event_id(self) -> None:
        response = self.client.post(
            f"/api/agent/runs/{self.run_id}/event",
            json={
                "event_type": "status",
                "provider_id": "anthropic",
                "app_event": {"stage": "planning"},
            },
        )
        self.assertEqual(response.status_code, 200)
        event = response.json()["event"]
        # Defaults from Pydantic must produce None columns, not crashes.
        self.assertIsNone(event["call_id"])
        self.assertIsNone(event["parent_event_id"])
        events = self._fetch_events()
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["call_id"])
        self.assertIsNone(events[0]["parent_event_id"])


if __name__ == "__main__":
    unittest.main()
