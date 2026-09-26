"""Canvas session tokens — issuance, scope, expiry, audit (T-I1-09).

Everything here is a test of the one claim the design rests on: a credential
that lives in browser JavaScript is worth *one agent's graph for fifteen
minutes*, and not a run, a secret, or another agent.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from audit import audit_store  # noqa: E402
from canvas import canvas_access  # noqa: E402
from canvas.canvas_access import get_canvas_session_manager  # noqa: E402
from core import database  # noqa: E402
from pairing import pairing_service as pairing_service_module  # noqa: E402
from routes import agents as agents_routes  # noqa: E402
from routes import canvas_api  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

KERNEL_PRESENT = importlib.util.find_spec("agent_flow_core") is not None

# Cloudflare marks tunnel traffic with these; ``routes/deps.py`` reads them.
TUNNEL_HEADERS = {"CF-Connecting-IP": "203.0.113.9", "CF-Ray": "test-ray"}


class _StubPairing:
    """A pairing service that accepts exactly one key."""

    def __init__(self, paired_key: str) -> None:
        self._paired_key = paired_key

    def validate_api_key(self, api_key: str) -> bool:
        return api_key == self._paired_key


class CanvasSessionTokenTest(unittest.TestCase):
    PAIRED_KEY = "paired-key-for-tests"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "canvas_token_test.db"
        agent_store._agent_store = None
        schedule_store._store = None
        audit_store._audit_store = None
        get_canvas_session_manager().clear()

        self._original_pairing = pairing_service_module._pairing_service
        pairing_service_module._pairing_service = _StubPairing(self.PAIRED_KEY)

        # The app under test: the canvas front plus the agent routes it
        # delegates into, so a graph written through the canvas is the same
        # stored workflow the phone would read.
        app = FastAPI()
        app.include_router(agents_routes.router)
        app.include_router(canvas_api.session_router)
        app.include_router(canvas_api.router)
        app.dependency_overrides[verify_api_key] = lambda: self.PAIRED_KEY
        self.app = app
        self.client = TestClient(app)

        # A second app with *no* override, used to prove a canvas token is
        # not an API key. Its ``verify_api_key`` is the real one.
        strict = FastAPI()
        strict.include_router(agents_routes.router)
        self.strict_client = TestClient(strict)

        self.agent = self._create_agent()

    def tearDown(self) -> None:
        pairing_service_module._pairing_service = self._original_pairing
        get_canvas_session_manager().clear()
        agent_store._agent_store = None
        schedule_store._store = None
        audit_store._audit_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------

    def _create_agent(self, name: str = "canvas-bot") -> dict:
        response = self.client.post(
            "/api/agent/agents",
            json={
                "name": name,
                "description": "graph under test",
                "system_prompt": "You are useful.",
                "provider_id": "openai",
                "flow_json": [
                    {"id": "step_one", "name": "Plan", "type": "llm"},
                    {"id": "step_two", "name": "Report", "type": "llm"},
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _issue(self, agent_id: str | None = None) -> str:
        response = self.client.post(
            "/api/agent/canvas/session",
            json={"agent_id": agent_id or self.agent["id"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["token"]

    def _auth(self, token: str) -> dict[str, str]:
        return {canvas_api.CANVAS_TOKEN_HEADER: token}

    def _audit_events(self) -> list[dict]:
        return audit_store.get_audit_store().list_events(limit=200)

    # -- issuance ----------------------------------------------------------

    def test_issue_returns_a_scoped_short_lived_token(self) -> None:
        response = self.client.post(
            "/api/agent/canvas/session", json={"agent_id": self.agent["id"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["agent_id"], self.agent["id"])
        self.assertEqual(body["expires_in_minutes"], canvas_access.CANVAS_TOKEN_TTL_MINUTES)
        self.assertEqual(body["scope"], ["graph:read", "graph:write"])
        self.assertEqual(body["token_header"], canvas_api.CANVAS_TOKEN_HEADER)
        self.assertTrue(body["token"])

    def test_issue_requires_a_valid_api_key(self) -> None:
        def reject() -> str:
            raise HTTPException(status_code=401, detail="Invalid API key")

        self.app.dependency_overrides[verify_api_key] = reject
        try:
            response = self.client.post(
                "/api/agent/canvas/session", json={"agent_id": self.agent["id"]}
            )
        finally:
            self.app.dependency_overrides[verify_api_key] = lambda: self.PAIRED_KEY
        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(get_canvas_session_manager().active_count(), 0)

    def test_issue_refuses_an_agent_that_does_not_exist(self) -> None:
        response = self.client.post(
            "/api/agent/canvas/session", json={"agent_id": "agent_missing"}
        )
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(get_canvas_session_manager().active_count(), 0)

    # -- the token is not an API key ---------------------------------------

    def test_canvas_token_is_rejected_as_an_api_key(self) -> None:
        """Acceptance ①: the token buys nothing on ``/api/agent/*``."""
        token = self._issue()
        for path, method in (
            (f"/api/agent/agents/{self.agent['id']}", "get"),
            (f"/api/agent/agents/{self.agent['id']}/run-once", "post"),
        ):
            with self.subTest(path=path):
                response = self.strict_client.request(
                    method.upper(), path, headers={"X-API-Key": token}, json={}
                )
                self.assertEqual(response.status_code, 401, response.text)

    def test_the_paired_key_still_works_on_the_agent_routes(self) -> None:
        """Control for the test above: the 401 is about the token, not the app."""
        response = self.strict_client.get(
            f"/api/agent/agents/{self.agent['id']}",
            headers={"X-API-Key": self.PAIRED_KEY},
        )
        self.assertEqual(response.status_code, 200, response.text)

    # -- presenting the token ----------------------------------------------

    def test_missing_token_is_401(self) -> None:
        response = self.client.get(f"/api/canvas/agents/{self.agent['id']}/graph")
        self.assertEqual(response.status_code, 401, response.text)

    def test_unknown_token_is_401(self) -> None:
        response = self.client.get(
            f"/api/canvas/agents/{self.agent['id']}/graph",
            headers=self._auth("not-a-real-token"),
        )
        self.assertEqual(response.status_code, 401, response.text)

    def test_token_in_the_query_string_is_refused_with_an_explanation(self) -> None:
        token = self._issue()
        for key in ("token", "canvas_token", "api_key"):
            with self.subTest(key=key):
                response = self.client.get(
                    f"/api/canvas/agents/{self.agent['id']}/graph?{key}={token}",
                    headers=self._auth(token),
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(
                    canvas_api.CANVAS_TOKEN_HEADER, response.json()["detail"]
                )

    def test_expired_token_is_401(self) -> None:
        token = self._issue()
        manager = get_canvas_session_manager()
        session = manager.resolve(token).session
        assert session is not None
        expired = canvas_access.CanvasSession(
            token=session.token,
            agent_id=session.agent_id,
            created_at=session.created_at - timedelta(minutes=30),
            expires_at=datetime.now() - timedelta(seconds=1),
            api_key=session.api_key,
        )
        manager._sessions[token] = expired

        response = self.client.get(
            f"/api/canvas/agents/{self.agent['id']}/graph", headers=self._auth(token)
        )
        self.assertEqual(response.status_code, 401, response.text)
        self.assertIn("expired", response.json()["detail"])
        # And it is gone, not merely refused.
        self.assertIsNone(manager.resolve(token).session)

    def test_token_for_another_agent_is_403(self) -> None:
        """Acceptance ②: scope is one agent, not 'any agent'."""
        other = self._create_agent(name="other-bot")
        token = self._issue(self.agent["id"])
        response = self.client.get(
            f"/api/canvas/agents/{other['id']}/graph", headers=self._auth(token)
        )
        self.assertEqual(response.status_code, 403, response.text)

    # -- reads -------------------------------------------------------------

    def test_graph_read_returns_the_graph_and_nothing_wider(self) -> None:
        token = self._issue()
        response = self.client.get(
            f"/api/canvas/agents/{self.agent['id']}/graph", headers=self._auth(token)
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["id"], self.agent["id"])
        self.assertTrue("flow_graph" in body or "flow_graph_unavailable" in body)
        # A graph:read token asked for a graph.
        for wider in ("system_prompt", "tools_json", "policy_overrides_json", "flow_json"):
            self.assertNotIn(wider, body)

    def test_step_schema_is_reachable_with_a_canvas_token(self) -> None:
        token = self._issue()
        response = self.client.get(
            "/api/canvas/workflow/step-schema", headers=self._auth(token)
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("types", response.json())

    def test_option_source_scripts_is_reachable(self) -> None:
        token = self._issue()
        response = self.client.get(
            "/api/canvas/option-sources/scripts", headers=self._auth(token)
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["name"], "scripts")
        self.assertIn("scripts", body)

    def test_option_source_devices_delegates_to_the_devices_route(self) -> None:
        token = self._issue()
        with mock.patch.object(
            canvas_api.devices_routes,
            "list_devices",
            new=mock.AsyncMock(return_value={"devices": []}),
        ):
            response = self.client.get(
                "/api/canvas/option-sources/devices", headers=self._auth(token)
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"name": "devices", "devices": []})

    def test_unknown_option_source_is_404(self) -> None:
        token = self._issue()
        response = self.client.get(
            "/api/canvas/option-sources/secrets", headers=self._auth(token)
        )
        self.assertEqual(response.status_code, 404, response.text)

    # -- writes ------------------------------------------------------------

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_graph_round_trip_writes_through_the_existing_flow_graph_path(self) -> None:
        token = self._issue()
        read = self.client.get(
            f"/api/canvas/agents/{self.agent['id']}/graph", headers=self._auth(token)
        ).json()
        self.assertIn("flow_graph", read)

        response = self.client.patch(
            f"/api/canvas/agents/{self.agent['id']}/graph",
            headers=self._auth(token),
            json={"flow_graph": read["flow_graph"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("flow_graph", response.json())

        # The stored workflow is what the phone would now read.
        stored = self.strict_client.get(
            f"/api/agent/agents/{self.agent['id']}",
            headers={"X-API-Key": self.PAIRED_KEY},
        ).json()
        self.assertEqual(len(stored["flow_json"]), 2)

    def test_a_refused_graph_is_passed_through_verbatim(self) -> None:
        """Acceptance ⑥: the canvas shows the server's refusal, not a summary."""
        token = self._issue()
        response = self.client.patch(
            f"/api/canvas/agents/{self.agent['id']}/graph",
            headers=self._auth(token),
            json={"flow_graph": {"not": "a flow"}},
        )
        self.assertIn(response.status_code, (400, 422), response.text)
        body = response.json()
        # Either the kernel is absent (422 kernel_not_installed) or it is
        # present and refuses the shape (400 invalid_flow_graph). Both are
        # named refusals from ``routes/agents.py``; neither is invented here.
        self.assertIn(
            body["error"], {"kernel_not_installed", "invalid_flow_graph"}
        )
        self.assertTrue(body["message"])

    def test_write_with_a_token_for_another_agent_is_403(self) -> None:
        other = self._create_agent(name="third-bot")
        token = self._issue(self.agent["id"])
        response = self.client.patch(
            f"/api/canvas/agents/{other['id']}/graph",
            headers=self._auth(token),
            json={"flow_graph": {}},
        )
        self.assertEqual(response.status_code, 403, response.text)

    # -- external switch ---------------------------------------------------

    def test_tunnel_requests_are_403_when_the_switch_is_closed(self) -> None:
        token = self._issue()
        with mock.patch.object(
            canvas_api,
            "get_config",
            return_value=SimpleNamespace(canvas_external_enabled=False),
        ):
            blocked = self.client.get(
                f"/api/canvas/agents/{self.agent['id']}/graph",
                headers={**self._auth(token), **TUNNEL_HEADERS},
            )
            local = self.client.get(
                f"/api/canvas/agents/{self.agent['id']}/graph",
                headers=self._auth(token),
            )
        self.assertEqual(blocked.status_code, 403, blocked.text)
        self.assertEqual(local.status_code, 200, local.text)

    def test_tunnel_requests_are_allowed_by_default(self) -> None:
        token = self._issue()
        response = self.client.get(
            f"/api/canvas/agents/{self.agent['id']}/graph",
            headers={**self._auth(token), **TUNNEL_HEADERS},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_issuance_also_honours_the_closed_switch(self) -> None:
        with mock.patch.object(
            canvas_api,
            "get_config",
            return_value=SimpleNamespace(canvas_external_enabled=False),
        ):
            response = self.client.post(
                "/api/agent/canvas/session",
                json={"agent_id": self.agent["id"]},
                headers=TUNNEL_HEADERS,
            )
        self.assertEqual(response.status_code, 403, response.text)

    # -- audit -------------------------------------------------------------

    def test_audit_records_the_session_without_the_token_value(self) -> None:
        """Acceptance ⑦: a token never reaches disk."""
        token = self._issue()
        self.client.get(
            f"/api/canvas/agents/{self.agent['id']}/graph", headers=self._auth(token)
        )
        self.client.patch(
            f"/api/canvas/agents/{self.agent['id']}/graph",
            headers=self._auth(token),
            json={"flow_graph": {"not": "a flow"}},
        )

        events = self._audit_events()
        operations = {event["operation"] for event in events}
        self.assertIn("canvas.session.issue", operations)
        self.assertIn("canvas.graph.read", operations)
        self.assertIn("canvas.graph.write", operations)

        fingerprint = canvas_access.token_fingerprint(token)
        saw_fingerprint = False
        for event in events:
            serialized = repr(event)
            self.assertNotIn(token, serialized, f"token leaked into {event['operation']}")
            details = (event.get("payload") or {}).get("details") or {}
            if details.get("token_fingerprint") == fingerprint:
                saw_fingerprint = True
        self.assertTrue(saw_fingerprint, "no audit row could be tied to the session")

    def test_audit_records_a_rejected_token_with_its_reason(self) -> None:
        self.client.get(
            f"/api/canvas/agents/{self.agent['id']}/graph",
            headers=self._auth("bogus-token-value"),
        )
        events = self._audit_events()
        rejections = [e for e in events if e["operation"] == "canvas.token.rejected"]
        self.assertTrue(rejections)
        details = (rejections[0].get("payload") or {}).get("details") or {}
        self.assertEqual(details["reason"], "unknown")
        self.assertNotIn("bogus-token-value", repr(events))

    def test_audit_records_an_agent_mismatch(self) -> None:
        other = self._create_agent(name="fourth-bot")
        token = self._issue(self.agent["id"])
        self.client.get(
            f"/api/canvas/agents/{other['id']}/graph", headers=self._auth(token)
        )
        reasons = {
            ((e.get("payload") or {}).get("details") or {}).get("reason")
            for e in self._audit_events()
            if e["operation"] == "canvas.token.rejected"
        }
        self.assertIn("agent_mismatch", reasons)


class CanvasSessionManagerTest(unittest.TestCase):
    """The store itself, without HTTP."""

    def setUp(self) -> None:
        self.manager = canvas_access.CanvasSessionManager(ttl_minutes=15)

    def test_tokens_are_unguessable_and_unique(self) -> None:
        tokens = {self.manager.issue("agent_a").token for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        for token in tokens:
            self.assertGreaterEqual(len(token), 32)

    def test_expired_sessions_are_swept_on_issue(self) -> None:
        stale = self.manager.issue("agent_a")
        self.manager._sessions[stale.token] = canvas_access.CanvasSession(
            token=stale.token,
            agent_id=stale.agent_id,
            created_at=stale.created_at,
            expires_at=datetime.now() - timedelta(minutes=1),
        )
        self.manager.issue("agent_b")
        self.assertNotIn(stale.token, self.manager._sessions)
        self.assertEqual(self.manager.active_count(), 1)

    def test_scope_is_read_and_write_on_the_graph_only(self) -> None:
        session = self.manager.issue("agent_a")
        self.assertTrue(session.allows("graph:read"))
        self.assertTrue(session.allows("graph:write"))
        self.assertFalse(session.allows("run:start"))

    def test_fingerprint_is_stable_short_and_not_the_token(self) -> None:
        session = self.manager.issue("agent_a")
        self.assertEqual(session.fingerprint, session.fingerprint)
        self.assertEqual(len(session.fingerprint), 12)
        self.assertNotIn(session.fingerprint, session.token)

    def test_no_ip_keyed_ambient_session_exists(self) -> None:
        """Deliberately unlike ``PreviewAccessManager``.

        Preview binds a session to a client IP taken from ``X-Forwarded-For``,
        a header the client writes. Behind a tunnel that is not an identity,
        so the canvas has no such thing and this asserts it stays that way.
        """
        for forbidden in ("bind_remote_session", "has_remote_session", "get_client_ip"):
            self.assertFalse(hasattr(self.manager, forbidden), forbidden)


if __name__ == "__main__":
    unittest.main()
