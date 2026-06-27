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
    _headless,
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


class BrowserRuntimeManagerTest(unittest.TestCase):
    def test_headful_browser_defaults_to_offscreen_window(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _browser_launch_args(
                headless=False,
                viewport_width=1280,
                viewport_height=720,
            )

        self.assertIn("--window-size=1280,720", args)
        self.assertIn("--window-position=-32000,-32000", args)

    def test_visible_window_mode_leaves_browser_window_visible(self):
        with patch.dict(os.environ, {"CODEBRIDGE_BROWSER_WINDOW": "visible"}, clear=True):
            args = _browser_launch_args(
                headless=False,
                viewport_width=1280,
                viewport_height=720,
            )

        self.assertEqual(args, [])

    def test_headless_browser_does_not_set_window_position(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _browser_launch_args(
                headless=True,
                viewport_width=1280,
                viewport_height=720,
            )

        self.assertEqual(args, [])

    def test_browser_runtime_is_headless_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_headless())

    def test_headless_can_be_disabled_for_debugging(self):
        with patch.dict(os.environ, {"CODEBRIDGE_BROWSER_HEADLESS": "0"}, clear=True):
            self.assertFalse(_headless())

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
