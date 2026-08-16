"""The three answers that used to be hardcoded, and what each one costs.

`browser_action` steps used to launch bundled Chromium, headless, with no
profile — three decisions taken in code, none of them askable. The product
promises "check the site every morning", and a browser with no profile cannot
log in, so the promise could not be kept.

What is asserted here is the shape of the choice, not a policy:

*   an unconfigured install still runs, with the documented defaults,
*   an installed Chrome is preferred and costs no download,
*   the option that hands the agent every login the operator's own Chrome
    holds is never arrived at by default or by fallback — only by choosing it,
    and it states its consequence when chosen,
*   an unsatisfiable choice is reported, not quietly replaced by another one.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from core import database  # noqa: E402
from system import browser_preferences as prefs  # noqa: E402
from system.browser_runtime_setup import InstalledBrowser  # noqa: E402

CHROME = InstalledBrowser(
    channel="chrome",
    name="Google Chrome",
    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    user_data_dir="/Users/someone/Library/Application Support/Google/Chrome",
)
CHROME_NO_PROFILE = InstalledBrowser(
    channel="chrome",
    name="Google Chrome",
    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    user_data_dir=None,
)


def plan(preferences=None, *, chrome=None, chromium_present=None):
    return prefs.resolve_browser_launch_plan(
        preferences or prefs.BrowserPreferences(),
        chrome=chrome,
        detect_chrome=False,
        chromium_present=chromium_present,
    )


class DefaultsTest(unittest.TestCase):
    def test_an_unconfigured_install_has_stated_defaults(self):
        defaults = prefs.BrowserPreferences()
        self.assertEqual(defaults.browser, prefs.BROWSER_AUTO)
        self.assertEqual(defaults.headless_mode, prefs.HEADLESS_AUTO)
        self.assertEqual(defaults.profile, prefs.PROFILE_DEDICATED)

    def test_the_default_profile_is_the_agents_own_never_the_operators_chrome(self):
        resolved = plan(chrome=CHROME)
        self.assertEqual(resolved.profile, prefs.PROFILE_DEDICATED)
        self.assertNotEqual(resolved.user_data_dir, CHROME.user_data_dir)
        self.assertTrue(resolved.persistent, "a login must survive to the next run")

    def test_auto_is_headless(self):
        self.assertTrue(plan(chrome=CHROME).headless)


class BrowserChoiceTest(unittest.TestCase):
    def test_installed_chrome_is_preferred_and_needs_no_download(self):
        resolved = plan(chrome=CHROME, chromium_present=False)
        self.assertEqual(resolved.browser, prefs.BROWSER_CHROME)
        self.assertEqual(resolved.channel, "chrome")
        self.assertFalse(resolved.install_required)
        self.assertIn("Chrome", resolved.label)

    def test_without_chrome_auto_falls_back_to_bundled_chromium(self):
        resolved = plan(chrome=None, chromium_present=False)
        self.assertEqual(resolved.browser, prefs.BROWSER_CHROMIUM)
        self.assertIsNone(resolved.channel)
        self.assertTrue(resolved.install_required, "this is the machine that pays 200MB")

    def test_chromium_present_needs_no_install(self):
        resolved = plan(chrome=None, chromium_present=True)
        self.assertFalse(resolved.install_required)

    def test_asking_for_chrome_on_a_machine_without_it_is_refused_not_swapped(self):
        resolved = plan(
            prefs.BrowserPreferences(browser=prefs.BROWSER_CHROME), chrome=None
        )
        self.assertFalse(resolved.usable)
        self.assertEqual(resolved.blocked_reason, prefs.BLOCKED_CHROME_NOT_INSTALLED)

    def test_asking_for_chromium_ignores_an_installed_chrome(self):
        resolved = plan(
            prefs.BrowserPreferences(browser=prefs.BROWSER_CHROMIUM), chrome=CHROME
        )
        self.assertEqual(resolved.browser, prefs.BROWSER_CHROMIUM)
        self.assertIsNone(resolved.channel)


class ProfileChoiceTest(unittest.TestCase):
    def test_sharing_the_operators_chrome_states_both_costs(self):
        resolved = plan(
            prefs.BrowserPreferences(profile=prefs.PROFILE_SHARED_CHROME),
            chrome=CHROME,
        )
        self.assertTrue(resolved.usable)
        self.assertEqual(resolved.user_data_dir, CHROME.user_data_dir)
        codes = {warning["code"] for warning in resolved.warnings}
        self.assertIn(prefs.WARN_SHARED_PROFILE_SCOPE, codes)
        self.assertIn(prefs.WARN_SHARED_PROFILE_LOCK, codes)

    def test_sharing_needs_chrome_and_says_so_rather_than_using_chromium(self):
        resolved = plan(
            prefs.BrowserPreferences(
                browser=prefs.BROWSER_CHROMIUM, profile=prefs.PROFILE_SHARED_CHROME
            ),
            chrome=CHROME,
        )
        self.assertFalse(resolved.usable)
        self.assertEqual(
            resolved.blocked_reason, prefs.BLOCKED_SHARED_PROFILE_NEEDS_CHROME
        )

    def test_a_missing_chrome_profile_directory_is_reported_not_substituted(self):
        resolved = plan(
            prefs.BrowserPreferences(profile=prefs.PROFILE_SHARED_CHROME),
            chrome=CHROME_NO_PROFILE,
        )
        self.assertFalse(resolved.usable)
        self.assertEqual(resolved.blocked_reason, prefs.BLOCKED_SHARED_PROFILE_MISSING)
        # The failure mode this prevents: silently running in the agent's own
        # profile, signed out, while the operator believes they shared theirs.
        self.assertNotEqual(resolved.user_data_dir, str(prefs.dedicated_profile_dir()))

    def test_no_profile_keeps_nothing(self):
        resolved = plan(
            prefs.BrowserPreferences(profile=prefs.PROFILE_EPHEMERAL), chrome=CHROME
        )
        self.assertFalse(resolved.persistent)
        self.assertIsNone(resolved.user_data_dir)


class HeadedTest(unittest.TestCase):
    def test_headed_is_honoured_and_names_what_it_needs(self):
        resolved = plan(
            prefs.BrowserPreferences(headless_mode=prefs.HEADLESS_OFF), chrome=CHROME
        )
        self.assertFalse(resolved.headless)
        self.assertIn(
            prefs.WARN_HEADED_NEEDS_DISPLAY,
            {warning["code"] for warning in resolved.warnings},
        )

    def test_headless_is_honoured(self):
        resolved = plan(
            prefs.BrowserPreferences(headless_mode=prefs.HEADLESS_ON), chrome=CHROME
        )
        self.assertTrue(resolved.headless)


class StorageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "browser_prefs.db"
        database._settings_db = None
        database.init_db()

    def tearDown(self):
        database._settings_db = None
        database.DB_PATH = self._original
        self._tmp.cleanup()

    def test_a_stored_choice_round_trips(self):
        prefs.set_browser_preferences(
            browser=prefs.BROWSER_CHROME,
            headless_mode=prefs.HEADLESS_OFF,
            profile=prefs.PROFILE_SHARED_CHROME,
        )
        stored = prefs.get_browser_preferences()
        self.assertEqual(stored.browser, prefs.BROWSER_CHROME)
        self.assertEqual(stored.headless_mode, prefs.HEADLESS_OFF)
        self.assertEqual(stored.profile, prefs.PROFILE_SHARED_CHROME)

    def test_a_partial_update_leaves_the_other_answers_alone(self):
        prefs.set_browser_preferences(profile=prefs.PROFILE_EPHEMERAL)
        prefs.set_browser_preferences(headless_mode=prefs.HEADLESS_OFF)
        stored = prefs.get_browser_preferences()
        self.assertEqual(stored.profile, prefs.PROFILE_EPHEMERAL)
        self.assertEqual(stored.headless_mode, prefs.HEADLESS_OFF)

    def test_an_unknown_value_is_refused_rather_than_rounded_off(self):
        with self.assertRaises(ValueError):
            prefs.set_browser_preferences(profile="my_chrome")
        self.assertEqual(
            prefs.get_browser_preferences().profile, prefs.PROFILE_DEDICATED
        )

    def test_an_unreadable_store_falls_back_to_defaults_instead_of_crashing(self):
        with mock.patch.object(
            database, "get_settings_db", side_effect=RuntimeError("db is gone")
        ):
            stored = prefs.get_browser_preferences()
        self.assertEqual(stored.profile, prefs.PROFILE_DEDICATED)

    def test_corrupt_stored_values_do_not_leak_into_a_launch(self):
        database.get_settings_db().set_json(
            prefs.SETTING_KEY, {"profile": "shared_everything", "browser": 17}
        )
        stored = prefs.get_browser_preferences()
        self.assertEqual(stored.profile, prefs.PROFILE_DEDICATED)
        self.assertEqual(stored.browser, prefs.BROWSER_AUTO)


class OptionCatalogTest(unittest.TestCase):
    def test_every_implemented_value_is_offered_with_a_description(self):
        options = prefs.preference_options()
        self.assertEqual(
            [item["value"] for item in options["browser"]], list(prefs.BROWSER_CHOICES)
        )
        self.assertEqual(
            [item["value"] for item in options["headless_mode"]],
            list(prefs.HEADLESS_MODES),
        )
        self.assertEqual(
            [item["value"] for item in options["profile"]], list(prefs.PROFILE_MODES)
        )
        for group in options.values():
            for item in group:
                self.assertTrue(item["label"])
                self.assertTrue(item["detail"])

    def test_the_shared_profile_option_says_what_it_gives_away(self):
        shared = next(
            item
            for item in prefs.preference_options()["profile"]
            if item["value"] == prefs.PROFILE_SHARED_CHROME
        )
        self.assertEqual(shared.get("consequence"), "shared")
        self.assertIn("every site", shared["detail"])


if __name__ == "__main__":
    unittest.main()
