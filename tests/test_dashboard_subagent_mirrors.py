"""The dashboard must be able to reach every subagent endpoint the phone can.

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
the next subagent route remembers the second registration.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import app_factory  # noqa: E402

SHARED_PREFIX = "/api/agent/subagents"
DASHBOARD_PREFIX = "/api/dashboard/agent/subagents"


class DashboardSubagentMirrorTest(unittest.TestCase):
    def setUp(self):
        dashboard = app_factory.create_dashboard_app()
        self.paths = {getattr(route, "path", "") for route in dashboard.routes}

    def test_every_shared_subagent_route_has_a_dashboard_mirror(self):
        shared = sorted(p for p in self.paths if p.startswith(SHARED_PREFIX))
        self.assertTrue(shared, "precondition: shared subagent routes exist")

        for path in shared:
            mirrored = DASHBOARD_PREFIX + path[len(SHARED_PREFIX) :]
            with self.subTest(path=path):
                self.assertIn(
                    mirrored,
                    self.paths,
                    f"{path} has no dashboard mirror at {mirrored}; the "
                    "dashboard will 404 and render it as an empty result",
                )


if __name__ == "__main__":
    unittest.main()
