"""A site's refusal must reach the run, and typing must reach the editor.

Both were found while posting to a Naver cafe through the product path.

The post never appeared, yet the step reported `completed`:

  * `fill` sets a DOM value. A rich-text editor keeps its own document model
    and updates it from input events, so the title (a plain textarea) took the
    text and the body stayed empty.
  * The site then refused the empty post with an `alert`. Playwright dismisses
    dialogs by default — which is what keeps an unattended run from hanging —
    so the refusal, and the site's own words for it, were discarded.

Together those two turned "nothing was posted" into a successful-looking run,
which is the failure mode worth pinning: not a crash, a false success.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import _attach_dialog_recorder  # noqa: E402


class _Dialog:
    def __init__(self, message: str) -> None:
        self.message = message


class _Page:
    """Minimal stand-in for a Playwright page's event surface."""

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}

    def on(self, event: str, handler) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, payload) -> None:
        for handler in self._handlers.get(event, []):
            handler(payload)


class DialogRecorderTest(unittest.TestCase):
    def test_a_refusal_is_captured_with_the_sites_own_words(self) -> None:
        page, seen = _Page(), []
        _attach_dialog_recorder(page, seen)
        page.emit("dialog", _Dialog("내용을 입력해주세요."))
        self.assertEqual(seen, ["내용을 입력해주세요."])

    def test_several_dialogs_all_survive(self) -> None:
        page, seen = _Page(), []
        _attach_dialog_recorder(page, seen)
        page.emit("dialog", _Dialog("first"))
        page.emit("dialog", _Dialog("second"))
        self.assertEqual(seen, ["first", "second"])

    def test_a_page_without_events_is_not_an_error(self) -> None:
        """Fake pages in tests have no `.on`; a missing hook must not take the
        step down — it only means there are no dialogs to record."""

        class _Bare:
            pass

        seen: list[str] = []
        _attach_dialog_recorder(_Bare(), seen)
        self.assertEqual(seen, [])


class TypeUsesRealKeystrokesTest(unittest.IsolatedAsyncioTestCase):
    """`type` and `fill` must not be the same thing.

    They were, and that is why the cafe body stayed empty: `fill` writes the
    value, `type` produces the input events an editor listens for.
    """

    async def test_type_clicks_then_sends_keystrokes(self) -> None:
        from agent.browser_action_adapter import PlaywrightBrowserActionAdapter

        source = Path(
            PlaywrightBrowserActionAdapter.__module__.replace(".", "/") + ".py"
        )
        text = (SERVER_DIR / source).read_text(encoding="utf-8")
        self.assertIn(
            "keyboard.type",
            text,
            "`type` must send real keystrokes; without them a rich-text editor "
            "receives nothing and the submit is rejected for empty content",
        )
        self.assertIn('if action_type == "type":', text)


if __name__ == "__main__":
    unittest.main()
