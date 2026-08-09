"""A client that can grant a permission must be able to look up what it means.

The phone can write standing rules: `POST /api/policies/rules` is on the
shared router behind the api key. For a while the catalogs describing what
those operations govern were dashboard-only, on the reasoning that they were
"UI metadata the phone does not need". The result was a phone that could
grant `provider.tool` — every tool the model asks for beyond bash, edit and
git — while having no reachable way to discover that is what it means.

Reading a catalog grants nothing; it describes the runtime's own gating. So
the rule this pins is one-directional and narrow: wherever rule *writing* is
reachable, operation *lookup* must be reachable too. If someone later moves
the catalog back behind the dashboard, or opens rule writing on a new surface
without the catalog, this fails rather than shipping a client that asks people
to authorise words it cannot define.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import app_factory  # noqa: E402

RULES = "/api/policies/rules"
CATALOGS = ("/api/policies/operations", "/api/policies/step-operations")


def _paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


class PolicyCatalogReachabilityTest(unittest.TestCase):
    def setUp(self):
        self.api = _paths(app_factory.create_api_app())
        self.dashboard = _paths(app_factory.create_dashboard_app())

    def test_the_phone_can_look_up_what_it_is_allowed_to_grant(self):
        self.assertIn(RULES, self.api, "precondition: the phone writes rules")
        for path in CATALOGS:
            with self.subTest(path=path):
                self.assertIn(
                    path,
                    self.api,
                    f"{path} is unreachable from the phone, which can still "
                    f"POST {RULES} — that asks someone to authorise an "
                    "operation whose meaning they cannot fetch",
                )

    def test_the_dashboard_keeps_its_own_prefixed_mirror(self):
        """Moving the catalog must not break the page that already used it."""
        for path in (
            "/api/dashboard/agent/policies/operations",
            "/api/dashboard/agent/policies/step-operations",
        ):
            with self.subTest(path=path):
                self.assertIn(path, self.dashboard)

    def test_the_catalogs_are_read_only(self):
        """Reading describes the gate; it must never be a way through it.

        If a write verb ever appears on a catalog path, the argument for
        exposing it to the phone no longer holds.
        """
        for app, label in (
            (app_factory.create_api_app(), "api"),
            (app_factory.create_dashboard_app(), "dashboard"),
        ):
            for route in app.routes:
                path = getattr(route, "path", "")
                if not path.endswith(("/policies/operations", "/policies/step-operations")):
                    continue
                methods = set(getattr(route, "methods", set()) or set())
                with self.subTest(app=label, path=path):
                    self.assertTrue(
                        methods <= {"GET", "HEAD", "OPTIONS"},
                        f"{path} on the {label} app exposes {methods}",
                    )


if __name__ == "__main__":
    unittest.main()
