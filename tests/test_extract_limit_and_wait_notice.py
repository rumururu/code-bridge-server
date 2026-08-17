"""Two quiet gaps left by the browser work, both about what goes unsaid.

`extract` kept the first 10,000 characters of a page and said nothing about the
rest. Reading a cafe's board listing hit that cap, and a step whose job is
"read the page and summarise it" received half a page with no sign it was half.
A cap is right — this text is stored with the run — but a silent one is not.

The wait notification told the user an agent needs a login. It did not say that
it is the only notification coming: the throttle holds the next one for 24h
(the user's own "at most once per day"), and until someone acts every scheduled
run of that agent parks and is suppressed. Saying so costs no extra push.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import (  # noqa: E402
    _EXTRACT_DEFAULT_CHARS,
    _EXTRACT_MAX_CHARS,
    _extract,
    _extract_char_limit,
)


class _Loc:
    def __init__(self, text: str) -> None:
        self._text = text
        self.first = self

    async def inner_text(self, timeout=None):
        return self._text

    async def get_attribute(self, name, timeout=None):
        return None


class _Page:
    def __init__(self, text: str) -> None:
        self._text = text

    def locator(self, selector):
        return _Loc(self._text)


class ExtractLimitTest(unittest.IsolatedAsyncioTestCase):
    def test_the_default_is_used_when_the_step_says_nothing(self) -> None:
        self.assertEqual(_extract_char_limit({}), _EXTRACT_DEFAULT_CHARS)

    def test_a_step_can_ask_for_more(self) -> None:
        self.assertEqual(_extract_char_limit({"max_chars": 50_000}), 50_000)

    def test_the_ceiling_holds(self) -> None:
        """The text is stored with the run; an unbounded page must not be."""
        self.assertEqual(
            _extract_char_limit({"max_chars": 10_000_000}), _EXTRACT_MAX_CHARS
        )

    def test_a_nonsense_limit_falls_back_rather_than_crashing(self) -> None:
        self.assertEqual(_extract_char_limit({"max_chars": "lots"}), _EXTRACT_DEFAULT_CHARS)
        self.assertEqual(_extract_char_limit({"max_chars": 0}), 1)

    async def test_truncation_is_declared_not_hidden(self) -> None:
        page = _Page("x" * (_EXTRACT_DEFAULT_CHARS + 500))
        found = await _extract(page, {"type": "extract"}, index=1)
        self.assertTrue(found["truncated"])
        self.assertEqual(found["full_length"], _EXTRACT_DEFAULT_CHARS + 500)

    async def test_a_page_that_fits_is_not_marked_truncated(self) -> None:
        found = await _extract(_Page("short"), {"type": "extract"}, index=1)
        self.assertNotIn("truncated", found)

    async def test_a_raised_limit_actually_keeps_the_rest(self) -> None:
        page = _Page("y" * 40_000)
        found = await _extract(
            page, {"type": "extract", "max_chars": 40_000}, index=1
        )
        self.assertEqual(len(found["text"]), 40_000)
        self.assertNotIn("truncated", found)

    async def test_a_named_value_is_never_lost_to_the_cap(self) -> None:
        """The pattern runs against the whole document. A value sitting past
        the cap must still bind, or raising the limit becomes a prerequisite
        for every extraction."""
        page = _Page("z" * (_EXTRACT_DEFAULT_CHARS + 100) + "clubid=31245773")
        found = await _extract(
            page,
            {"type": "extract", "name": "cafe_id", "pattern": r"clubid=(\d+)"},
            index=1,
        )
        self.assertEqual(found["value"], "31245773")


class WaitNotificationSaysWhatTheSilenceMeansTest(unittest.TestCase):
    def _body(self, reason: str, body: str = "다시 로그인해 주세요."):
        from agent.task_orchestrator import _wait_notification_body

        return _wait_notification_body(body, reason=reason)

    def test_a_login_park_says_the_agent_stays_stopped(self) -> None:
        text = self._body("login_required")
        self.assertIn("다시 로그인해 주세요.", text)
        self.assertIn("24시간", text)

    def test_a_bot_check_says_the_same(self) -> None:
        self.assertIn("24시간", self._body("captcha_or_bot_challenge"))

    def test_an_ordinary_ask_is_left_alone(self) -> None:
        """A step asking a question does not block the agent, so the warning
        would be wrong there."""
        self.assertEqual(self._body("ask_user"), "다시 로그인해 주세요.")


if __name__ == "__main__":
    unittest.main()
