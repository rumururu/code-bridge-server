"""A browser workflow must be able to use what the page told it.

Workflows are authored before they ever run, so they cannot contain the values
a site only reveals at runtime. A Naver cafe is the case that exposed this: the
address a person types is `cafe.naver.com/devsharing`, but the board listing
lives at `/f-e/cafes/31245773/menus/0`, and that number appears nowhere except
inside the page. Before this, `extract` could read the page but nothing could
consume the result, so the only way to reach the listing was to hardcode one
cafe's id — a workflow that works for exactly one cafe and no other.

The unbound case matters just as much: a `{{name}}` nobody filled must still
park the run and ask, never quietly resolve to an empty string and navigate
somewhere arbitrary while reporting success.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import (  # noqa: E402
    _requires_user_target,
    bind_action,
)


class BindActionTest(unittest.TestCase):
    def test_a_runtime_value_reaches_a_later_url(self) -> None:
        action = {
            "type": "navigate",
            "url": "https://cafe.naver.com/f-e/cafes/{{cafe_id}}/menus/0",
        }
        bound = bind_action(action, {"cafe_id": "31245773"})
        self.assertEqual(
            bound["url"], "https://cafe.naver.com/f-e/cafes/31245773/menus/0"
        )

    def test_selectors_and_typed_text_bind_too(self) -> None:
        bound = bind_action(
            {"type": "fill", "selector": "#row-{{idx}}", "text": "re: {{title}}"},
            {"idx": "3", "title": "품앗이"},
        )
        self.assertEqual(bound["selector"], "#row-3")
        self.assertEqual(bound["text"], "re: 품앗이")

    def test_an_unbound_name_survives_and_still_parks_the_run(self) -> None:
        action = {"type": "navigate", "url": "{{cafe_id}}"}
        bound = bind_action(action, {"other": "x"})
        self.assertEqual(bound["url"], "{{cafe_id}}")
        self.assertTrue(
            _requires_user_target(bound, "navigate"),
            "an unfilled reference must ask, not navigate to an empty URL and "
            "call the step done",
        )

    def test_a_reference_inside_a_longer_url_still_parks(self) -> None:
        """Measured failure: `.../cafes/{{cafe_id}}/menus/0` was not a whole-
        string placeholder, so the run navigated to the braces percent-encoded
        (`/cafes/%7B%7Bcafe_id%7D%7D/menus/0`), got a page, and reported
        `completed`. An unfilled reference must stop the run wherever it sits."""
        action = {
            "type": "navigate",
            "url": "https://cafe.naver.com/f-e/cafes/{{cafe_id}}/menus/0",
        }
        bound = bind_action(action, {})
        self.assertTrue(_requires_user_target(bound, "navigate"))

    def test_a_bound_action_no_longer_asks(self) -> None:
        bound = bind_action(
            {"type": "navigate", "url": "https://x.test/{{id}}"}, {"id": "7"}
        )
        self.assertFalse(_requires_user_target(bound, "navigate"))

    def test_the_original_action_is_not_mutated(self) -> None:
        """Observations record the action as authored; rewriting it in place
        would make the run record disagree with the workflow."""
        action = {"type": "navigate", "url": "https://x.test/{{id}}"}
        bind_action(action, {"id": "7"})
        self.assertEqual(action["url"], "https://x.test/{{id}}")

    def test_fields_outside_the_bindable_set_are_left_alone(self) -> None:
        action = {"type": "navigate", "url": "https://x.test", "note": "{{id}}"}
        self.assertEqual(bind_action(action, {"id": "7"})["note"], "{{id}}")

    def test_no_bindings_is_a_no_op(self) -> None:
        action = {"type": "navigate", "url": "https://x.test/{{id}}"}
        self.assertIs(bind_action(action, {}), action)


class ExtractNamesValuesTest(unittest.IsolatedAsyncioTestCase):
    """`extract` is the only way a value enters the bindings."""

    async def _extract(self, action, *, text="", attribute_value=None, html=""):
        from agent.browser_action_adapter import _extract

        class _Loc:
            first = None

            async def inner_text(self, timeout=None):
                return text

            async def get_attribute(self, name, timeout=None):
                return attribute_value

        loc = _Loc()
        loc.first = loc

        class _Page:
            def locator(self, selector):
                return loc

            async def content(self):
                return html

        return await _extract(_Page(), action, index=1)

    async def test_an_identifier_is_found_in_markup_not_visible_text(self) -> None:
        """Measured failure: a cafe id lives in an href, so extracting the
        body's *visible text* matched nothing and bound nothing."""
        found = await self._extract(
            {
                "type": "extract",
                "name": "cafe_id",
                "source": "html",
                "pattern": r"clubid[=:\"\s]+(\d{6,})",
            },
            text="보이는 글자에는 id가 없다",
            html='<a href="/ArticleList.nhn?search.clubid=31245773">전체글보기</a>',
        )
        self.assertEqual(found["value"], "31245773")

    async def test_a_pattern_names_a_value_buried_in_the_page(self) -> None:
        found = await self._extract(
            {"type": "extract", "name": "cafe_id", "pattern": r"clubid=(\d+)"},
            text='...<a href="/ArticleList.nhn?search.clubid=31245773">...',
        )
        self.assertEqual(found["name"], "cafe_id")
        self.assertEqual(found["value"], "31245773")
        self.assertTrue(found["matched"])

    async def test_a_pattern_that_matches_nothing_binds_nothing(self) -> None:
        """Binding the whole page text under the name would send the next
        action to a nonsense URL instead of stopping."""
        found = await self._extract(
            {"type": "extract", "name": "cafe_id", "pattern": r"clubid=(\d+)"},
            text="no id here",
        )
        self.assertFalse(found["matched"])
        self.assertNotIn("value", found)

    async def test_an_attribute_can_be_captured(self) -> None:
        found = await self._extract(
            {
                "type": "extract",
                "name": "cafe_url",
                "selector": "a.cafe",
                "attribute": "href",
            },
            attribute_value="https://cafe.naver.com/devsharing",
        )
        self.assertEqual(found["value"], "https://cafe.naver.com/devsharing")

    async def test_extract_without_a_name_still_just_reads(self) -> None:
        found = await self._extract({"type": "extract"}, text="hello")
        self.assertEqual(found["text"], "hello")
        self.assertNotIn("name", found)


if __name__ == "__main__":
    unittest.main()
