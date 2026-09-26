"""The code that saved a dialog's words stopped the click that opened it.

Playwright auto-dismisses dialogs only while nothing is listening. Attaching a
`dialog` listener turns that off and makes the handler responsible; an
unhandled dialog stays on screen and the action that opened it never returns.

`_attach_dialog_recorder` took that responsibility — it was added so a submit
a site refused could not be reported as a success — and never discharged it.
Measured on a cafe post, the submit click logged

    - waiting for element to be visible, enabled and stable
    - element is visible, enabled and stable
    - scrolling into view if needed
    - done scrolling
    - performing click action

and then hung the full thirty seconds. The absence of `element does not
receive pointer events` or `element is not stable` is the tell: nothing was
covering the button and nothing was moving. The click ran, opened a dialog,
and waited for an answer that no longer came from anywhere.
"""

from __future__ import annotations

import asyncio
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
        self.dismissed = False

    def dismiss(self) -> None:
        self.dismissed = True


class _AsyncDialog(_Dialog):
    async def dismiss(self) -> None:  # type: ignore[override]
        self.dismissed = True


class _Page:
    def __init__(self) -> None:
        self.handler = None

    def on(self, event: str, handler) -> None:
        assert event == "dialog"
        self.handler = handler


class DialogRecorderTests(unittest.TestCase):
    def test_the_message_is_still_recorded(self) -> None:
        """The reason the listener exists at all."""

        page, sink = _Page(), []
        _attach_dialog_recorder(page, sink)

        page.handler(_Dialog("제목을 입력해 주세요."))

        self.assertEqual(sink, ["제목을 입력해 주세요."])

    def test_the_dialog_is_settled_so_the_click_can_return(self) -> None:
        page, sink = _Page(), []
        _attach_dialog_recorder(page, sink)
        dialog = _Dialog("등록하시겠습니까?")

        page.handler(dialog)

        self.assertTrue(
            dialog.dismissed,
            "an unanswered dialog hangs the action that opened it",
        )

    def test_a_dialog_that_cannot_be_settled_still_yields_its_words(self) -> None:
        """A broken dismiss must not cost the message as well as the step."""

        class _Stuck(_Dialog):
            def dismiss(self) -> None:  # type: ignore[override]
                raise RuntimeError("target closed")

        page, sink = _Page(), []
        _attach_dialog_recorder(page, sink)

        page.handler(_Stuck("사라질 문장"))  # must not raise

        self.assertEqual(sink, ["사라질 문장"])

    def test_a_page_with_no_hook_is_not_an_error(self) -> None:
        """Fakes without `.on` are common in this suite."""

        _attach_dialog_recorder(object(), [])


class AsyncDialogTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_awaitable_dismiss_is_scheduled_not_dropped(self) -> None:
        page, sink = _Page(), []
        _attach_dialog_recorder(page, sink)
        dialog = _AsyncDialog("등록하시겠습니까?")

        page.handler(dialog)
        await asyncio.sleep(0)

        self.assertEqual(sink, ["등록하시겠습니까?"])
        self.assertTrue(dialog.dismissed)


if __name__ == "__main__":
    unittest.main()
