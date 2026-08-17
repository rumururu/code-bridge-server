"""An expired session must be noticed on any site, not on one.

The detection was the literal string `nid.naver.com/nidlogin`. Naver's expiry
was caught; every other site's was invisible — the step landed on a sign-in
form, read it as an ordinary page, and finished. A run that is silently signed
out but reports success is the failure this guards, and it cannot depend on
which site produced it.

The opposite mistake matters as much: a workflow whose whole job is to sign in
must not be parked for arriving at the login page it asked for. So the signal
is a sign-in page the step did **not** ask to be on.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import _blocked_state, _same_destination  # noqa: E402


class _Locator:
    def __init__(self, count: int = 0, visible: bool = False) -> None:
        self._count, self._visible = count, visible
        self.first = self

    async def count(self) -> int:
        return self._count

    async def is_visible(self) -> bool:
        return self._visible

    async def inner_text(self, timeout: int | None = None) -> str:
        return ""


class _Page:
    def __init__(self, url: str, *, password_field: bool = False, body: str = "") -> None:
        self.url = url
        self._password = password_field
        self._body = body

    def locator(self, selector: str):
        if "password" in selector:
            return _Locator(count=1 if self._password else 0, visible=self._password)
        loc = _Locator()

        async def _text(timeout: int | None = None) -> str:
            return self._body

        loc.inner_text = _text  # type: ignore[method-assign]
        return loc


class ExpiredSessionIsNoticedAnywhereTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_bounce_to_a_password_form_parks_the_run(self) -> None:
        page = _Page(
            "https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fcafe.naver.com",
            password_field=True,
        )
        self.assertEqual(
            await _blocked_state(
                page, requested_url="https://cafe.naver.com/f-e/cafes/1/menus/0"
            ),
            "login_required",
        )

    async def test_a_site_that_is_not_naver_is_noticed_too(self) -> None:
        """The whole point: no site's name appears in the rule."""
        page = _Page("https://accounts.example.com/signin?next=%2Fdash", password_field=True)
        self.assertEqual(
            await _blocked_state(page, requested_url="https://app.example.com/dash"),
            "login_required",
        )

    async def test_an_identifier_first_page_with_no_password_field(self) -> None:
        """Some sign-ins ask for the identifier first and show no password box
        yet. The diverted URL's wording is what is left to go on."""
        page = _Page("https://accounts.example.com/o/oauth2/auth?client_id=x")
        self.assertEqual(
            await _blocked_state(page, requested_url="https://app.example.com/reports"),
            "login_required",
        )


class DeliberateSignInIsNotParkedTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_flow_that_meant_to_open_the_login_page_runs_on(self) -> None:
        """Parking here would break every workflow whose job is to sign in."""
        url = "https://nid.naver.com/nidlogin.login?mode=form"
        self.assertIsNone(await _blocked_state(_Page(url, password_field=True), requested_url=url))

    async def test_an_ordinary_page_is_not_a_login(self) -> None:
        page = _Page("https://cafe.naver.com/f-e/cafes/1/menus/0")
        self.assertIsNone(
            await _blocked_state(page, requested_url="https://cafe.naver.com/f-e/cafes/1/menus/0")
        )

    async def test_a_password_field_on_the_page_you_asked_for_is_not_a_bounce(self) -> None:
        """A settings page can hold a password field. Arriving where the step
        pointed is not an expiry, whatever the page contains."""
        url = "https://app.example.com/settings/security"
        self.assertIsNone(await _blocked_state(_Page(url, password_field=True), requested_url=url))


class TheDestinationIsCarriedAcrossActionsTest(unittest.IsolatedAsyncioTestCase):
    """The check runs after every action, not only after a navigation.

    Measured: a flow that deliberately opened a login page and then ran a
    `wait` parked on the wait — the wait carries no URL of its own, so the
    login page looked like somewhere nobody had asked to be. The step's last
    stated destination has to persist between actions.
    """

    LOGIN = "https://nid.naver.com/nidlogin.login?mode=form"

    async def test_a_wait_after_a_deliberate_login_visit_does_not_park(self) -> None:
        page = _Page(self.LOGIN, password_field=True)
        self.assertIsNone(await _blocked_state(page, requested_url=self.LOGIN))

    async def test_with_no_destination_yet_the_page_alone_decides(self) -> None:
        """A resumed run may not have navigated at all; then a password form is
        all there is to go on, and parking is the safe answer."""
        page = _Page(self.LOGIN, password_field=True)
        self.assertEqual(await _blocked_state(page, requested_url=None), "login_required")


class SameDestinationTest(unittest.TestCase):
    def test_a_fragment_or_trailing_slash_is_not_a_redirect(self) -> None:
        self.assertTrue(_same_destination("https://a.test/x", "https://a.test/x#top"))
        self.assertTrue(_same_destination("https://a.test/x/", "https://a.test/x"))

    def test_a_query_string_is_significant(self) -> None:
        """`…/nidlogin.login?url=<where you were going>` is what a bounce looks
        like, so the query cannot be discarded."""
        self.assertFalse(
            _same_destination("https://a.test/x", "https://a.test/login?url=%2Fx")
        )


class CaptchaStillDetectedTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_bot_check_still_parks(self) -> None:
        page = _Page("https://a.test/x", body="보안문자를 입력해 주세요")
        self.assertEqual(
            await _blocked_state(page, requested_url="https://a.test/x"),
            "captcha_or_bot_challenge",
        )


if __name__ == "__main__":
    unittest.main()
