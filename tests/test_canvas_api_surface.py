"""The permanent gate on how much a leaked canvas token is worth — T-I1-09.

A canvas token lives in browser JavaScript, so it *will* be readable by
whatever else runs on that page. The design does not pretend otherwise; it
bounds the damage instead, and the bound is literally a list of four routes.

The request that eventually arrives is "let me hit Run from the canvas".
Granting it silently would turn every leaked canvas token into shell execution
on the operator's machine — ``shell`` steps, the secret store, the filesystem.
So the list is frozen here. If this test goes red, the surface grew: decide
that on purpose, in review, and update the snapshot in the same commit that
argues for it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import FastAPI

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.workflow_step_schema import OPTION_SOURCES  # noqa: E402
from routes import canvas_api, register_api_routers, register_dashboard_routers  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

# The whole of what a canvas token can reach.
CANVAS_ROUTE_SNAPSHOT = {
    ("GET", "/api/canvas/agents/{agent_id}/graph"),
    ("PATCH", "/api/canvas/agents/{agent_id}/graph"),
    ("GET", "/api/canvas/workflow/step-schema"),
    ("GET", "/api/canvas/option-sources/{name}"),
}

# Issuance is api-key-gated and lives outside the canvas prefix on purpose, so
# it cannot be reached with a canvas token and cannot inflate the list above.
CANVAS_SESSION_ROUTE_SNAPSHOT = {("POST", "/api/agent/canvas/session")}

# Words that name capabilities a graph editor must never be handed.
FORBIDDEN_IN_CANVAS_PATHS = (
    "run",
    "dry-run",
    "shell",
    "terminal",
    "script",
    "secret",
    "builder",
    "file",
    "filesystem",
    "git",
    "approval",
    "chat",
)


def _routes_with_prefix(app: FastAPI, prefix: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith(prefix):
            continue
        for method in getattr(route, "methods", set()) or set():
            if method in {"HEAD", "OPTIONS"}:
                continue
            found.add((method, path))
    return found


class CanvasApiSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api_app = FastAPI()
        register_api_routers(self.api_app)
        self.dashboard_app = FastAPI()
        register_dashboard_routers(self.dashboard_app)

    def test_canvas_prefix_holds_exactly_four_routes(self) -> None:
        for name, app in (("api", self.api_app), ("dashboard", self.dashboard_app)):
            with self.subTest(app=name):
                self.assertEqual(
                    _routes_with_prefix(app, "/api/canvas"),
                    CANVAS_ROUTE_SNAPSHOT,
                )

    def test_session_issuance_is_one_route_outside_the_canvas_prefix(self) -> None:
        self.assertEqual(
            _routes_with_prefix(self.api_app, "/api/agent/canvas"),
            CANVAS_SESSION_ROUTE_SNAPSHOT,
        )

    def test_no_canvas_route_names_an_execution_capability(self) -> None:
        for method, path in _routes_with_prefix(self.api_app, "/api/canvas"):
            tail = path[len("/api/canvas") :].lower()
            for word in FORBIDDEN_IN_CANVAS_PATHS:
                with self.subTest(path=path, word=word):
                    self.assertNotIn(word, tail, f"{method} {path} names '{word}'")

    def test_canvas_routes_are_gated_by_the_canvas_token_only(self) -> None:
        """No canvas route may accept an API key, and none may be ungated."""
        for route in self.api_app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/api/canvas"):
                continue
            calls = {
                dependency.call
                for dependency in route.dependant.dependencies
            }
            with self.subTest(path=path):
                self.assertIn(canvas_api.verify_canvas_token, calls)
                self.assertNotIn(verify_api_key, calls)

    def test_session_issuance_is_gated_by_a_real_api_key(self) -> None:
        route = next(
            r for r in self.api_app.routes
            if getattr(r, "path", "") == "/api/agent/canvas/session"
        )
        calls = {dependency.call for dependency in route.dependant.dependencies}
        self.assertIn(verify_api_key, calls)
        self.assertNotIn(canvas_api.verify_canvas_token, calls)

    def test_every_published_option_source_has_a_resolver(self) -> None:
        """Drift guard: the schema's list and the canvas's list are one list.

        ``workflow_step_schema`` refuses to publish an option source with
        nothing behind it. The canvas is that "behind" for the browser, so a
        new source added there without a resolver here would leave the canvas
        with a silently empty dropdown.
        """
        self.assertEqual(
            set(canvas_api._OPTION_SOURCE_RESOLVERS),
            set(OPTION_SOURCES),
        )


if __name__ == "__main__":
    unittest.main()
