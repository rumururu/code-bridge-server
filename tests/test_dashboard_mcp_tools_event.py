"""The agents page must be able to show what `task.step.mcp_tools` recorded.

C6-2's contract is that a declared-but-undetected MCP server does not vanish
silently: `_apply_declared_mcp_servers` (agent/task_orchestrator.py) writes a
`task.step.mcp_tools` run event naming every server it injected, every one
that runs on this server's own runtime, every one missing from this machine,
and every one it could not pass through. Until this page consumed it, that
event went into `agent_events` and stopped — stored, streamed, and shown to
nobody, which is only half the contract.

Two things are asserted here, same shape as the browser-runtime chip tests:
the dashboard can actually read a run's events (the shared route is
api-key-gated and the dashboard has no key), and the page renders the event
from the stored payload rather than inventing a state of its own.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import app_factory  # noqa: E402
from routes.deps import require_local_access  # noqa: E402


def _fake_store(events):
    store = MagicMock()
    store.get_run.return_value = {"id": "run_1"}
    store.list_events.return_value = events
    return store


class RunEventsMirrorTest(unittest.TestCase):
    def setUp(self):
        self.app = app_factory.create_dashboard_app()
        self.app.dependency_overrides[require_local_access] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_dashboard_mirror_is_registered(self):
        paths = {getattr(route, "path", "") for route in self.app.routes}
        self.assertIn("/api/dashboard/agent/runs/{run_id}/events", paths)

    def test_mirror_returns_the_same_events_the_phone_gets(self):
        events = [
            {
                "id": "evt_1",
                "run_id": "run_1",
                "sequence": 1,
                "event_type": "task.step.mcp_tools",
                "provider_id": "anthropic",
                "provider_event": None,
                "app_event": {
                    "task_id": "task_1",
                    "step_id": "step_1",
                    "agent_id": "agent_1",
                    "declared": ["playwright", "gmail"],
                    "injected": ["playwright"],
                    "builtin_runtime": [],
                    "missing": [{"mcp_id": "gmail", "detail": "not configured"}],
                    "not_injected": [],
                    "provider_supports_mcp_injection": True,
                    "message": "Passed to this run: playwright.",
                },
                "call_id": None,
                "parent_event_id": None,
                "created_at": "2026-08-16T00:00:00",
            }
        ]
        with patch(
            "routes.agents.get_agent_store", return_value=_fake_store(events)
        ):
            response = self.client.get("/api/dashboard/agent/runs/run_1/events")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"events": events})

    def test_mirror_passes_paging_through(self):
        store = _fake_store([])
        with patch("routes.agents.get_agent_store", return_value=store):
            response = self.client.get(
                "/api/dashboard/agent/runs/run_1/events?after_sequence=7&limit=9"
            )

        self.assertEqual(response.status_code, 200)
        store.list_events.assert_called_once_with(
            "run_1", after_sequence=7, limit=9
        )

    def test_unknown_run_is_a_404_not_an_empty_list(self):
        store = _fake_store([])
        store.get_run.return_value = None
        with patch("routes.agents.get_agent_store", return_value=store):
            response = self.client.get("/api/dashboard/agent/runs/run_x/events")

        self.assertEqual(response.status_code, 404)

    def test_api_app_does_not_gain_a_dashboard_prefixed_route(self):
        api_paths = {
            getattr(route, "path", "") for route in app_factory.create_api_app().routes
        }
        self.assertNotIn("/api/dashboard/agent/runs/{run_id}/events", api_paths)


class McpToolsTemplateTest(unittest.TestCase):
    def setUp(self):
        self.markup = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
            encoding="utf-8"
        )

    def test_run_detail_fetches_the_events_too(self):
        """The report is an event, not a step output — steps alone cannot show it."""
        self.assertIn("/events?limit=500", self.markup)

    def test_only_the_mcp_event_type_is_rendered_specially(self):
        self.assertIn("event.event_type === 'task.step.mcp_tools'", self.markup)
        self.assertIn("renderMcpToolsEvent", self.markup)

    def test_missing_servers_are_visibly_warned_in_both_locales(self):
        for locale_key in ("mcp_missing:", "mcp_injected:", "mcp_builtin:"):
            self.assertEqual(
                self.markup.count(locale_key),
                2,
                f"{locale_key} must be defined in both en and ko locale tables",
            )
        # The warning names the server rather than summarising it away.
        self.assertIn("t('mcp_missing').replace('{id}', entry.mcp_id)", self.markup)
        self.assertIn('mcp-line warn', self.markup)

    def test_not_injected_shows_the_servers_own_sentence(self):
        """The server already wrote why (unlaunchable config, or a session
        type with no MCP option); the page must show that sentence, not
        compose one that could contradict it."""
        self.assertIn("payload.not_injected", self.markup)
        self.assertIn("entry.detail", self.markup)

    def test_unmatched_events_fall_back_to_run_level_not_nowhere(self):
        self.assertIn("runLevel.push(event)", self.markup)

    def test_unrecognised_payload_renders_nothing_not_reassurance(self):
        self.assertIn(
            "if (!payload || typeof payload !== 'object') return '';", self.markup
        )


if __name__ == "__main__":
    unittest.main()
