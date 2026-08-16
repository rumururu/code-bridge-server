import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_runtime_manager import (  # noqa: E402
    LiveBrowserRuntime,
    _browser_launch_args,
    _headless_override,
    offscreen_window_args,
)
from system.browser_preferences import (  # noqa: E402
    BrowserPreferences,
    resolve_browser_launch_plan,
)


class _FakePage:
    def __init__(self):
        self.evaluations = []
        self.keyboard = _FakeKeyboard()

    async def evaluate(self, script, *args):
        self.evaluations.append((script, args))
        return True


class _FakeKeyboard:
    def __init__(self):
        self.pressed = []

    async def press(self, key):
        self.pressed.append(key)



def _clean_env(**overrides: str) -> dict[str, str]:
    """A bare environment that still keeps the suite off the operator's install.

    `patch.dict(os.environ, {}, clear=True)` is the right instinct — these tests
    are about what the browser does with *no* configuration — but clearing
    everything also removes the isolation `conftest.py` sets at import time.
    `system/browser_preferences.py:144` resolves the profile through
    `runtime_dir(...)`, so with `CODEBRIDGE_APP_SUPPORT_DIR` gone it falls back
    to `server/core/browser_profile` and a test run writes a real Chrome profile
    into the checkout — ~190 files that then show up as orphaned code on the
    next deployment. Keep the isolation, clear the rest.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "CODEBRIDGE_APP_SUPPORT_DIR",
            "CODEBRIDGE_FCM_SERVICE_ACCOUNT",
            "CODE_BRIDGE_SERVER_LOG_PATH",
        }
    }
    env.update(overrides)
    return env


class BrowserRuntimeManagerTest(unittest.TestCase):
    def test_headful_browser_defaults_to_offscreen_window(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            args = _browser_launch_args(
                headless=False,
                viewport_width=1280,
                viewport_height=720,
            )

        self.assertIn("--window-size=1280,720", args)
        self.assertIn("--window-position=-32000,-32000", args)

    def test_visible_window_mode_leaves_browser_window_visible(self):
        with patch.dict(os.environ, _clean_env(**{"CODEBRIDGE_BROWSER_WINDOW": "visible"}), clear=True):
            args = _browser_launch_args(
                headless=False,
                viewport_width=1280,
                viewport_height=720,
            )

        self.assertEqual(args, [])

    def test_headless_browser_does_not_set_window_position(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            args = _browser_launch_args(
                headless=True,
                viewport_width=1280,
                viewport_height=720,
            )

        self.assertEqual(args, [])

    def test_offscreen_args_are_shared_with_the_workflow_adapter(self):
        # The adapter's headed launch must position its window the same way the
        # handoff does; two copies of "-32000" would drift.
        with patch.dict(os.environ, _clean_env(), clear=True):
            self.assertEqual(
                offscreen_window_args(viewport_width=1280, viewport_height=720),
                _browser_launch_args(
                    headless=False, viewport_width=1280, viewport_height=720
                ),
            )

    def test_unset_env_means_no_override_so_the_stored_setting_decides(self):
        # The env var used to default to "1", which silently overrode the
        # operator's stored choice on every machine — a headed setting could
        # never take effect. Unset now means "no opinion".
        with patch.dict(os.environ, _clean_env(), clear=True):
            self.assertIsNone(_headless_override())

    def test_headless_is_still_the_effective_default(self):
        # No override and no stored preference: a scheduled run has nobody
        # watching and a server may have no display, so it stays headless.
        with patch.dict(os.environ, _clean_env(), clear=True):
            plan = resolve_browser_launch_plan(
                BrowserPreferences(), detect_chrome=False
            )
        self.assertTrue(plan.headless)

    def test_headless_can_be_disabled_for_debugging(self):
        with patch.dict(os.environ, _clean_env(**{"CODEBRIDGE_BROWSER_HEADLESS": "0"}), clear=True):
            self.assertFalse(_headless_override())

    def test_reset_scroll_to_top_scrolls_live_page_to_origin(self):
        page = _FakePage()
        runtime = LiveBrowserRuntime(
            {
                "id": "bs_test",
                "context_dir": "/tmp/bs_test",
                "status": "waiting_for_user",
            },
            playwright=object(),
            browser=object(),
            context=object(),
            page=page,
            viewport_width=1280,
            viewport_height=720,
            headless=True,
        )

        import asyncio

        asyncio.run(runtime.reset_scroll_to_top())

        self.assertEqual(
            page.evaluations,
            [("() => { if (window && window.scrollTo) window.scrollTo(0, 0); }", ())],
        )

    def test_scroll_page_by_uses_direct_page_scroll(self):
        page = _FakePage()
        runtime = LiveBrowserRuntime(
            {
                "id": "bs_test",
                "context_dir": "/tmp/bs_test",
                "status": "waiting_for_user",
            },
            playwright=object(),
            browser=object(),
            context=object(),
            page=page,
            viewport_width=1280,
            viewport_height=720,
            headless=True,
        )

        import asyncio

        asyncio.run(runtime.scroll_page_by(650))

        script, args = page.evaluations[0]
        self.assertIn("window.scrollBy(0, deltaY)", script)
        self.assertIn("document.elementFromPoint", script)
        self.assertEqual(args, (650.0,))

    def test_clear_active_input_clears_focused_editable(self):
        page = _FakePage()
        runtime = LiveBrowserRuntime(
            {
                "id": "bs_test",
                "context_dir": "/tmp/bs_test",
                "status": "waiting_for_user",
            },
            playwright=object(),
            browser=object(),
            context=object(),
            page=page,
            viewport_width=1280,
            viewport_height=720,
            headless=True,
        )

        import asyncio

        asyncio.run(runtime.clear_active_input())

        script, args = page.evaluations[0]
        self.assertIn("document.activeElement", script)
        self.assertIn("el.value = \"\"", script)
        self.assertIn("InputEvent", script)
        self.assertEqual(args, ())
        self.assertEqual(page.keyboard.pressed, [])

if __name__ == "__main__":
    unittest.main()
