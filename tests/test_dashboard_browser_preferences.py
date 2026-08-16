"""Where the operator answers the three questions, and who is allowed to.

The choice lives on the dashboard, localhost-only, for the same reason the
Chromium install trigger does: one of the options hands the agent every login
the person's own Chrome holds. `agents_router` is shared with the
tunnel-exposed app, so a leaked pairing key must not be enough to point someone
else's agent at their own browser sessions. The phone still sees the *result*
through the readiness probe — it just cannot change it.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import app_factory  # noqa: E402
from core import database  # noqa: E402
from routes.deps import require_local_access  # noqa: E402
from system import browser_preferences as prefs  # noqa: E402

PATH = "/api/dashboard/agent/browser-runtime/preferences"


class BrowserPreferencesRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "dashboard_browser_prefs.db"
        database._settings_db = None
        database.init_db()
        self.app = app_factory.create_dashboard_app()
        self.app.dependency_overrides[require_local_access] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        database._settings_db = None
        database.DB_PATH = self._original_db
        self._tmp.cleanup()

    def test_get_returns_the_current_answers_and_every_option(self):
        payload = self.client.get(PATH).json()
        self.assertEqual(payload["preferences"]["profile"], prefs.PROFILE_DEDICATED)
        values = [item["value"] for item in payload["options"]["profile"]]
        self.assertEqual(values, list(prefs.PROFILE_MODES))
        self.assertIn("label", payload["plan"])

    def test_the_shared_option_is_offered_with_its_consequence_attached(self):
        options = self.client.get(PATH).json()["options"]["profile"]
        shared = next(item for item in options if item["value"] == prefs.PROFILE_SHARED_CHROME)
        self.assertEqual(shared.get("consequence"), "shared")
        self.assertIn("every site", shared["detail"])

    def test_put_stores_the_choice_and_re_answers_readiness(self):
        response = self.client.put(
            PATH,
            json={"browser": prefs.BROWSER_CHROMIUM, "headless_mode": prefs.HEADLESS_OFF},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["preferences"]["browser"], prefs.BROWSER_CHROMIUM)
        self.assertEqual(payload["preferences"]["headless_mode"], prefs.HEADLESS_OFF)
        # The answer that was cached a moment ago was about a different
        # browser, so a fresh one has to come back with the write.
        self.assertIn("ready", payload["readiness"])
        self.assertEqual(
            prefs.get_browser_preferences().browser, prefs.BROWSER_CHROMIUM
        )

    def test_an_unimplemented_value_is_refused_with_400(self):
        response = self.client.put(PATH, json={"profile": "my_own_chrome_but_safe"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            prefs.get_browser_preferences().profile, prefs.PROFILE_DEDICATED
        )

    def test_an_empty_body_changes_nothing(self):
        self.client.put(PATH, json={"profile": prefs.PROFILE_EPHEMERAL})
        self.client.put(PATH, json={})
        self.assertEqual(
            prefs.get_browser_preferences().profile, prefs.PROFILE_EPHEMERAL
        )

    def test_the_trigger_is_localhost_only_and_never_on_the_tunnel(self):
        api_paths = {
            getattr(route, "path", "") for route in app_factory.create_api_app().routes
        }
        self.assertNotIn(PATH, api_paths)
        for path in api_paths:
            self.assertNotIn("browser-runtime/preferences", path)


class BrowserPreferencesTemplateTest(unittest.TestCase):
    def setUp(self):
        self.markup = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
            encoding="utf-8"
        )

    def test_the_page_loads_and_saves_the_preferences(self):
        self.assertIn("api('/browser-runtime/preferences')", self.markup)
        self.assertIn("'/browser-runtime/preferences', {", self.markup)

    def test_the_settings_are_reachable_when_the_runtime_is_already_working(self):
        # The chip used to do nothing when green, which would have hidden the
        # settings from exactly the person whose browser runtime works and who
        # wants to change whose logins it uses.
        self.assertIn("alwaysOpens: true", self.markup)

    def test_a_machine_that_needs_no_download_is_not_offered_one(self):
        self.assertIn("browserRuntime.install_required !== false", self.markup)

    def test_the_wording_exists_in_both_languages(self):
        for key in (
            "modal_browser_prefs_title",
            "modal_browser_will_use",
            "modal_browser_pref_profile",
        ):
            self.assertEqual(
                self.markup.count(f"{key}:"), 2, f"{key} must exist in en and ko"
            )

    def test_the_shared_profile_option_is_translated_and_states_its_scope(self):
        self.assertIn("shared_chrome: {", self.markup)
        self.assertIn("모든 사이트에 접근할 수 있습니다", self.markup)


if __name__ == "__main__":
    unittest.main()
