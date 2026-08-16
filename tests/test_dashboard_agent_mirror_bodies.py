"""The dashboard mirrors must accept the same request bodies the phone does.

`routes/dashboard_agents.py` re-exposes agent operations for the localhost
dashboard by delegating to the handlers in `routes/agents.py`. That delegation
is only safe while both sides agree on the request model.

They stopped agreeing when the commit gate landed. `routes/agents.py` grew
route-level subclasses (`BuilderCommitBody`, `AgentCreateBody`,
`AgentUpdateBody`) carrying `commit_incomplete`, and its handlers read
`body.commit_incomplete` directly — but the mirrors still declared the plain
`BuilderCommitRequest` / `AgentCreate` / `AgentUpdate`. FastAPI validated the
request against the mirror's annotation, so the object handed to the handler
had no such attribute and every dashboard create/update/commit raised
`AttributeError` — HTTP 500 — while the identical call from the phone
succeeded, because the phone reaches `routes/agents.py` directly.

The whole server suite was green when that shipped: the tests exercised the
`/api/agent/*` routes, and nothing asserted that the two paths take the same
body. These tests do, structurally, so the next field added to a Body subclass
cannot silently break the dashboard again.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routes import agents as agents_routes  # noqa: E402
from routes import dashboard_agents  # noqa: E402


def _body_annotation(func, param_name: str = "body"):
    """The declared type of a route's request body, following __wrapped__."""
    return func.__annotations__[param_name]


class DashboardMirrorsTakeTheRouteLevelBodies(unittest.TestCase):
    """Each mirror must declare the same body model as the handler it calls."""

    MIRRORED = (
        ("create_agent", "create_agent"),
        ("update_agent", "update_agent"),
        ("builder_commit", "builder_commit"),
    )

    def test_mirror_body_matches_the_handler_it_delegates_to(self):
        for mirror_name, handler_name in self.MIRRORED:
            with self.subTest(route=mirror_name):
                mirror = getattr(dashboard_agents, mirror_name)
                handler = getattr(agents_routes, handler_name)
                self.assertIs(
                    _body_annotation(mirror),
                    _body_annotation(handler),
                    f"dashboard mirror {mirror_name!r} declares a different body "
                    f"model than routes.agents.{handler_name} reads from",
                )

    def test_every_field_the_handler_reads_survives_the_mirror(self):
        """The concrete failure: `commit_incomplete` must exist on what arrives.

        Asserting the field rather than only the class identity means a future
        refactor that re-splits the models still has to keep the escape hatch
        reachable from the dashboard.
        """
        for mirror_name, _ in self.MIRRORED:
            with self.subTest(route=mirror_name):
                model = _body_annotation(getattr(dashboard_agents, mirror_name))
                self.assertIn(
                    "commit_incomplete",
                    model.model_fields,
                    f"{model.__name__} drops commit_incomplete, so the dashboard "
                    "cannot answer the commit gate's refusal",
                )


if __name__ == "__main__":
    unittest.main()
