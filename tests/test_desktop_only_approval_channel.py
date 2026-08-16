"""``desktop_only`` must be decided by the channel, not by the request body.

The policy engine escalates the things it least wants approved casually —
``file.read`` of ``~/.ssh/id_rsa``, a workspace ``.env``,
``firebase_service_account.json``, ``AuthKey_*.p8`` — to ``desktop_only``,
which means "a person at this machine has to decide this".

Enforcement used to read ``approver["type"]`` out of the JSON body, and the
Flutter phone app sends ``{'type': 'desktop_app'}``
(``lib/providers/approval_provider.dart``). So the control was self-asserted:
a phone tap approved exactly the reads the escalation existed to keep off the
phone. These tests pin the replacement — the channel the request arrived on:

* ``/api/approvals/{id}/decision`` is api-key gated (``verify_api_key``), i.e.
  a paired client, possibly over a tunnel → never desktop.
* ``/api/dashboard/agent/approvals/{id}/decision`` sits behind
  ``require_local_access`` on a router registered only in
  ``_DASHBOARD_ONLY_ROUTERS``, so it exists solely on the localhost-bound
  dashboard listener → desktop.

Both doors are exercised here on purpose. The naive fix hardens one and either
leaves the other forgeable or demotes the dashboard so *nothing* can approve a
``desktop_only`` request — and this codebase has already shipped exactly that
class of mirror/route drift once (see test_dashboard_agent_mirror_bodies.py).
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from approvals import approval_store  # noqa: E402
from approvals.approval_models import ApprovalDecisionCreate  # noqa: E402
from approvals.approver_channel import (  # noqa: E402
    CHANNEL_DESKTOP,
    CHANNEL_REMOTE,
    is_desktop_channel,
    stamp_approver,
)
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402
from routes import approvals as approvals_routes  # noqa: E402
from routes import audit as audit_routes  # noqa: E402
from routes import dashboard_agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

PHONE_APPROVER = {"type": "desktop_app", "device_name": "Pixel 8", "id": "phone-1"}

REMOTE_DECISION_PATH = "/api/approvals/{approval_id}/decision"
DESKTOP_DECISION_PATH = "/api/dashboard/agent/approvals/{approval_id}/decision"


class DesktopOnlyApprovalChannelTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "desktop_only_channel_test.db"
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(approvals_routes.router)
        app.include_router(dashboard_agents.router)
        app.include_router(audit_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # -- helpers ----------------------------------------------------------

    def _request(self, operation: str, details: dict) -> dict:
        response = self.client.post(
            "/api/approvals/request",
            json={"operation": operation, "run_id": "run_demo", "details": details},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _pending_secret_read(self) -> dict:
        """A real escalation: reading the user's private SSH key."""
        payload = self._request("file.read", {"path": "~/.ssh/id_rsa"})
        self.assertTrue(payload["approval_required"])
        self.assertTrue(
            payload["approval"]["desktop_only"],
            "reading ~/.ssh/id_rsa should still be desktop_only",
        )
        return payload["approval"]

    def _pending_ordinary(self) -> dict:
        """An approval that is *not* desktop_only, for the control case."""
        payload = self._request("process.terminal", {"command": "npm install"})
        self.assertTrue(payload["approval_required"])
        self.assertFalse(payload["approval"]["desktop_only"])
        return payload["approval"]

    def _events(self) -> list[dict]:
        response = self.client.get("/api/audit/events?run_id=run_demo")
        self.assertEqual(response.status_code, 200)
        return response.json()["events"]

    # -- the defect -------------------------------------------------------

    def test_phone_claiming_desktop_app_is_refused_and_audited(self):
        approval = self._pending_secret_read()

        response = self.client.post(
            REMOTE_DECISION_PATH.format(approval_id=approval["id"]),
            json={"decision": "approve_once", "approver": PHONE_APPROVER},
        )

        self.assertEqual(response.status_code, 403)
        self.assertIn("Desktop-only", response.json()["error"])
        # Refused means still answerable by the desktop, not consumed.
        self.assertEqual(response.json()["approval"]["status"], "pending")

        rejected = [e for e in self._events() if e["decision"] == "approval_rejected"]
        self.assertEqual(len(rejected), 1, "the refusal must leave an audit trail")
        approver = rejected[0]["payload"]["approver"]
        # Device identity survives for the auditor...
        self.assertEqual(approver["device_name"], "Pixel 8")
        self.assertEqual(approver["id"], "phone-1")
        # ...but the claim is recorded as a claim, and the server's own view wins.
        self.assertEqual(approver["claimed_type"], "desktop_app")
        self.assertEqual(approver["channel"], CHANNEL_REMOTE)
        self.assertEqual(approver["type"], "remote_client")

    def test_dashboard_route_can_approve_the_same_request(self):
        approval = self._pending_secret_read()

        response = self.client.post(
            DESKTOP_DECISION_PATH.format(approval_id=approval["id"]),
            json={"decision": "approve_once", "approver": {"name": "PC dashboard"}},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["approval"]["status"], "approved")

        approved = [e for e in self._events() if e["decision"] == "approve_once"]
        self.assertEqual(len(approved), 1)
        approver = approved[0]["payload"]["approver"]
        self.assertEqual(approver["name"], "PC dashboard")
        self.assertEqual(approver["channel"], CHANNEL_DESKTOP)
        self.assertEqual(approver["type"], "desktop_app")

    def test_dashboard_route_is_not_demoted_by_a_remote_looking_body(self):
        """The body must not be able to *lower* the dashboard either.

        Deriving the channel from the request would be pointless if a stray
        ``approver.type`` could still steer it in the other direction and
        strand every desktop_only request with nobody able to answer it.
        """
        approval = self._pending_secret_read()

        response = self.client.post(
            DESKTOP_DECISION_PATH.format(approval_id=approval["id"]),
            json={
                "decision": "approve_once",
                "approver": {"type": "chat_banner", "channel": CHANNEL_REMOTE},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["approval"]["status"], "approved")

    def test_ordinary_approval_is_unaffected_from_both_routes(self):
        from_phone = self._pending_ordinary()
        response = self.client.post(
            REMOTE_DECISION_PATH.format(approval_id=from_phone["id"]),
            json={"decision": "approve_once", "approver": PHONE_APPROVER},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["approval"]["status"], "approved")

        from_desktop = self._pending_ordinary()
        response = self.client.post(
            DESKTOP_DECISION_PATH.format(approval_id=from_desktop["id"]),
            json={"decision": "approve_once"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["approval"]["status"], "approved")

    def test_phone_may_still_deny_a_desktop_only_request(self):
        """Refusing to approve is not refusing to answer.

        A denial lowers privilege, so the desktop restriction has no business
        blocking it — and blocking it would park the run forever.
        """
        approval = self._pending_secret_read()

        response = self.client.post(
            REMOTE_DECISION_PATH.format(approval_id=approval["id"]),
            json={"decision": "deny", "approver": PHONE_APPROVER},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["approval"]["status"], "denied")

    def test_every_desktop_only_escalation_is_covered_not_just_ssh(self):
        """The refusal is about the effect, not about one path pattern."""
        cases = {
            "settings.accessible_roots": {"path": "/Users/demo"},
            "file.read": {"path": "~/.aws/credentials"},
        }
        for operation, details in cases.items():
            with self.subTest(operation=operation):
                payload = self._request(operation, details)
                approval = payload["approval"]
                self.assertTrue(approval["desktop_only"])
                response = self.client.post(
                    REMOTE_DECISION_PATH.format(approval_id=approval["id"]),
                    json={"decision": "approve_once", "approver": PHONE_APPROVER},
                )
                self.assertEqual(response.status_code, 403)


class ApproverChannelUnitTest(unittest.TestCase):
    """The primitive itself, including what it refuses to trust."""

    def test_only_the_desktop_constant_counts_as_desktop(self):
        self.assertTrue(is_desktop_channel(CHANNEL_DESKTOP))
        for value in (
            CHANNEL_REMOTE,
            "desktop_app",
            "local_desktop_app",
            "DESKTOP ",  # tolerated: whitespace/case only
            None,
            "",
            "something_new",
        ):
            with self.subTest(channel=value):
                expected = str(value).strip().lower() == CHANNEL_DESKTOP if value else False
                self.assertEqual(is_desktop_channel(value), expected)

    def test_unknown_channel_fails_closed_to_remote(self):
        stamped = stamp_approver({"name": "mystery"}, channel="who_knows")
        self.assertEqual(stamped["channel"], CHANNEL_REMOTE)
        self.assertFalse(is_desktop_channel(stamped["channel"]))

    def test_client_cannot_seed_the_server_owned_keys(self):
        stamped = stamp_approver(
            {
                "type": "desktop_app",
                "channel": CHANNEL_DESKTOP,
                "claimed_type": "desktop_app",
                "device_name": "Pixel 8",
            },
            channel=CHANNEL_REMOTE,
        )
        self.assertEqual(stamped["channel"], CHANNEL_REMOTE)
        self.assertEqual(stamped["type"], "remote_client")
        self.assertEqual(stamped["claimed_type"], "desktop_app")
        self.assertEqual(stamped["device_name"], "Pixel 8")

    def test_missing_approver_still_yields_a_channel(self):
        stamped = stamp_approver(None, channel=CHANNEL_DESKTOP)
        self.assertEqual(stamped, {"type": "desktop_app", "channel": CHANNEL_DESKTOP})


class ChannelMustBeStatedAtEveryDoorTest(unittest.TestCase):
    """Structural guards, so the next route cannot inherit an answer."""

    def test_shared_handler_requires_an_explicit_channel(self):
        signature = inspect.signature(approvals_routes.apply_approval_decision)
        channel = signature.parameters["channel"]
        self.assertEqual(channel.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(
            channel.default,
            inspect.Parameter.empty,
            "channel must have no default here: a new call site has to say "
            "where its request came from, rather than silently inheriting one",
        )

    def test_each_door_declares_its_own_channel(self):
        """Both routes must reach the shared handler, each with its own channel.

        Asserted by intercepting the shared handler rather than by reading the
        source, so a mirror that quietly grew a second call path — the failure
        mode this codebase has already shipped once — shows up as a missing or
        wrong channel here instead of passing on a string match.
        """
        expected = {
            approvals_routes.create_approval_decision: CHANNEL_REMOTE,
            dashboard_agents.decide_approval: CHANNEL_DESKTOP,
        }
        body = ApprovalDecisionCreate(decision="approve_once")

        for route, channel in expected.items():
            with self.subTest(route=f"{route.__module__}.{route.__name__}"):
                seen: dict = {}

                async def _capture(approval_id, decision_body, *, channel, _seen=seen):
                    _seen["approval_id"] = approval_id
                    _seen["body"] = decision_body
                    _seen["channel"] = channel
                    return {"approval": None}

                with patch.object(approvals_routes, "apply_approval_decision", _capture):
                    asyncio.run(route("apv_1", body))

                self.assertEqual(
                    seen.get("channel"),
                    channel,
                    "route did not reach the shared handler with its own channel",
                )
                self.assertEqual(seen.get("approval_id"), "apv_1")
                self.assertIs(seen.get("body"), body)

    def test_the_desktop_router_is_never_mounted_on_the_api_app(self):
        """What the whole `CHANNEL_DESKTOP` claim rests on.

        The mirror is "the desktop" only because it exists solely on the
        localhost-bound dashboard listener. Moving its router into
        `_SHARED_ROUTERS` would put a desktop-trusted approval endpoint on the
        tunnel-exposed API app, and nothing else in the tree would complain.
        """
        import routes

        self.assertIn(routes.dashboard_agents_router, routes._DASHBOARD_ONLY_ROUTERS)
        self.assertNotIn(routes.dashboard_agents_router, routes._SHARED_ROUTERS)

        api_app = FastAPI()
        routes.register_api_routers(api_app)
        decision_paths = [
            route.path
            for route in api_app.routes
            if "approvals" in getattr(route, "path", "")
        ]
        self.assertTrue(decision_paths, "sanity: the api app should still have approvals")
        for path in decision_paths:
            self.assertNotIn(
                "/api/dashboard/",
                path,
                "a desktop-trusted approval route reached the tunnel-exposed app",
            )

    def test_service_default_channel_is_remote(self):
        from approvals.approval_service import decide_approval

        default = inspect.signature(decide_approval).parameters["channel"].default
        self.assertEqual(
            default,
            CHANNEL_REMOTE,
            "an in-process caller that says nothing must not be the desktop",
        )


if __name__ == "__main__":
    unittest.main()
