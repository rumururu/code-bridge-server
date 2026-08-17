"""A step must be able to assert where the site actually sent it.

Found while proving that a scheduled, unattended run is still signed in. The
check that matters is "we were not bounced to the login form" — and a bounce
looks like success by every other measure: HTTP 200, a real title, visible
text. The only difference is the URL.

`url_contains` did not exist, so the step raised `unsupported assert kind`,
failed, and its `on_failure: ask_user` parked the run — a scenario that cannot
be written rather than one that failed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import _assert_page  # noqa: E402


class _Page:
    def __init__(self, url: str) -> None:
        self.url = url


class UrlAssertionTest(unittest.IsolatedAsyncioTestCase):
    SIGNED_IN = "https://nid.naver.com/user2/help/myInfoV2"
    BOUNCED = "https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fnid.naver.com"

    async def test_still_signed_in_passes(self) -> None:
        await _assert_page(
            _Page(self.SIGNED_IN), {"kind": "url_contains", "value": "myInfoV2"}
        )

    async def test_a_bounce_to_the_login_form_fails(self) -> None:
        with self.assertRaises(AssertionError):
            await _assert_page(
                _Page(self.BOUNCED), {"kind": "url_contains", "value": "myInfoV2"}
            )

    async def test_the_negative_form_catches_the_same_bounce(self) -> None:
        """Written the other way round — "we must not be at a login page" — the
        signed-out URL is what has to fail."""
        await _assert_page(
            _Page(self.SIGNED_IN),
            {"kind": "url_not_contains", "value": "nidlogin"},
        )
        with self.assertRaises(AssertionError):
            await _assert_page(
                _Page(self.BOUNCED),
                {"kind": "url_not_contains", "value": "nidlogin"},
            )

    async def test_a_missing_value_is_rejected_rather_than_passing(self) -> None:
        """An empty needle is `"" in url` — true for every URL. Silently
        passing would make the assertion decorative."""
        with self.assertRaises(ValueError):
            await _assert_page(_Page(self.BOUNCED), {"kind": "url_contains"})

    async def test_the_failure_says_which_url_it_saw(self) -> None:
        with self.assertRaises(AssertionError) as caught:
            await _assert_page(
                _Page(self.BOUNCED), {"kind": "url_contains", "value": "myInfoV2"}
            )
        self.assertIn("nidlogin", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
