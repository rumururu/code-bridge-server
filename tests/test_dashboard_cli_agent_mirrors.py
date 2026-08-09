"""The dashboard must be able to reach every CLI-agent endpoint the phone can.

The dashboard talks to itself under `/api/dashboard/agent`. Shared routes are
also mounted on it under their own prefix, so a missing mirror is invisible
until someone calls the dashboard-prefixed path and gets a 404 — which is
exactly how the sweep mirror was missed: the feature shipped, the phone could
read it, and the dashboard could not, with nothing failing anywhere to say so.

A 404 here does not look like a bug either. It looks like an empty result:
the first thing that happened when the mirror was absent was a status page
rendering every field as blank, because the caller parsed the error body and
found no keys. "The server has not swept yet" and "you asked the wrong URL"
are very different statements, and only one of them was true.

So this asserts the pairing directly rather than trusting that whoever adds
the next route remembers the second registration.

The legacy `/subagents` prefix is held to the same rule. It is the prefix an
already-installed phone build calls, and the failure mode of dropping it is
the *same* one: not an error the user can see, but a screen that renders "you
have no agents".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import app_factory  # noqa: E402
from routes.cli_agents import (  # noqa: E402
    CLI_AGENTS_PREFIX,
    LEGACY_CLI_AGENTS_PREFIX,
)

DASHBOARD_ROOT = "/api/dashboard/agent"


class DashboardCliAgentMirrorTest(unittest.TestCase):
    def setUp(self):
        dashboard = app_factory.create_dashboard_app()
        self.paths = {getattr(route, "path", "") for route in dashboard.routes}

    def _assert_mirrored(self, shared_prefix: str) -> list[str]:
        shared = sorted(p for p in self.paths if p.startswith(shared_prefix))
        self.assertTrue(shared, f"precondition: {shared_prefix} routes exist")

        for path in shared:
            mirrored = DASHBOARD_ROOT + path[len("/api/agent") :]
            with self.subTest(path=path):
                self.assertIn(
                    mirrored,
                    self.paths,
                    f"{path} has no dashboard mirror at {mirrored}; the "
                    "dashboard will 404 and render it as an empty result",
                )
        return shared

    def test_every_shared_cli_agent_route_has_a_dashboard_mirror(self):
        self._assert_mirrored(CLI_AGENTS_PREFIX)

    def test_the_pre_rename_prefix_is_still_served_and_mirrored(self):
        """An installed phone build calls `/subagents` and must keep working.

        Dropping it would not raise anywhere — the client parses the 404 body,
        finds no `candidates` key, and shows an empty list.
        """
        legacy = self._assert_mirrored(LEGACY_CLI_AGENTS_PREFIX)

        canonical = {
            p[len(CLI_AGENTS_PREFIX) :]
            for p in self.paths
            if p.startswith(CLI_AGENTS_PREFIX)
        }
        self.assertEqual(
            {p[len(LEGACY_CLI_AGENTS_PREFIX) :] for p in legacy},
            canonical,
            "the legacy prefix must serve exactly the canonical route set — a "
            "route on one and not the other is a silent partial outage",
        )


if __name__ == "__main__":
    unittest.main()
