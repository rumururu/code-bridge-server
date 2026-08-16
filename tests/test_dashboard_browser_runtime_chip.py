"""The agents page must be able to say the browser runtime is missing.

A `browser_action` step on a server without Playwright does not fail — it
parks and waits for a person, forever if nobody is watching. The readiness
endpoint that knows this existed but nothing called it, so the page showed an
agent as fully ready right up until its scheduled run silently stopped.

Two things are asserted here: the dashboard can actually reach the probe
(the shared route is api-key-gated and the dashboard has no key), and the
page's chip is driven by the probe rather than assuming a default.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import app_factory  # noqa: E402
from routes.deps import require_local_access  # noqa: E402


class BrowserRuntimeMirrorTest(unittest.TestCase):
    def setUp(self):
        self.app = app_factory.create_dashboard_app()
        self.app.dependency_overrides[require_local_access] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_dashboard_mirror_is_registered(self):
        paths = {getattr(route, "path", "") for route in self.app.routes}
        self.assertIn("/api/dashboard/agent/browser-runtime/readiness", paths)

    def test_mirror_returns_the_same_probe_the_phone_gets(self):
        probe = {
            "ready": False,
            "playwright_python": False,
            "chromium_executable": False,
            "install_command": "python -m playwright install chromium",
            "message": "Python Playwright package is not available.",
            "diagnostics": [],
        }
        with patch(
            "routes.agents.get_browser_runtime_readiness",
            AsyncMock(return_value=probe),
        ):
            response = self.client.get("/api/dashboard/agent/browser-runtime/readiness")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), probe)

    def test_api_app_does_not_gain_a_dashboard_prefixed_route(self):
        api_paths = {
            getattr(route, "path", "") for route in app_factory.create_api_app().routes
        }
        self.assertNotIn("/api/dashboard/agent/browser-runtime/readiness", api_paths)


class BrowserChipTemplateTest(unittest.TestCase):
    def setUp(self):
        self.markup = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
            encoding="utf-8"
        )

    def test_page_fetches_the_readiness_probe(self):
        self.assertIn("api('/browser-runtime/readiness')", self.markup)

    def test_chip_only_appears_for_agents_with_a_browser_step(self):
        self.assertIn("step.type === 'browser_action'", self.markup)
        self.assertIn("key: 'browser'", self.markup)

    def test_chip_is_green_only_on_an_explicit_ready(self):
        """Unknown readiness must not render as ready."""
        self.assertIn("browserRuntime.ready === true", self.markup)

    def test_unknown_readiness_has_its_own_wording(self):
        self.assertIn("fix_browser_unknown", self.markup)
        for locale_key in ("r_browser:", "fix_browser:", "fix_browser_unknown:"):
            self.assertEqual(
                self.markup.count(locale_key),
                2,
                f"{locale_key} must be defined in both en and ko locale tables",
            )

    def test_install_command_is_offered_for_copying(self):
        self.assertIn("copyBrowserInstallCommand", self.markup)
        self.assertIn("browserRuntime.install_command", self.markup)


if __name__ == "__main__":
    unittest.main()
