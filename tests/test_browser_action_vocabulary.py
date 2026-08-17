"""The Configurator can only author actions it has been told exist.

The schema it was shown described `actions` as "a structured list (navigate,
click, type, ...)" and stopped there. So every browser step it produced was a
`navigate` to a placeholder URL plus a screenshot — a step that parks on its
first action. The vocabulary that makes a real browser workflow possible —
naming an extracted value and using it later, asserting the URL you landed on,
typing into a rich-text editor — was never visible to it.

Documenting a vocabulary creates a second problem: it can drift from what the
adapter actually dispatches, and a documented action that does not exist is
worse than an undocumented one, because the model will confidently emit it and
the step will stop with "unsupported browser action". These tests read the
adapter's own source and fail when the two disagree.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import (  # noqa: E402
    BROWSER_ACTION_VOCABULARY,
    BROWSER_ACTIONS_NOT_EXECUTED,
    browser_action_vocabulary_block,
)

ADAPTER_SOURCE = (SERVER_DIR / "agent" / "browser_action_adapter.py").read_text(
    encoding="utf-8"
)


def _dispatched_action_types() -> set[str]:
    """Every action type the adapter's run loop branches on."""
    loop = ADAPTER_SOURCE.split("for index, action in enumerate(actions", 1)[1]
    loop = loop.split("\n            finally:", 1)[0]
    found: set[str] = set()
    for match in re.finditer(r'action_type (?:==|in) (.+)', loop):
        found.update(re.findall(r'"([a-z_]+)"', match.group(1)))
    return found


class VocabularyMatchesTheAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documented = {name for name, _ in BROWSER_ACTION_VOCABULARY}
        self.dispatched = _dispatched_action_types()

    def test_the_loop_was_found(self) -> None:
        """Guard the guard: a refactor that moves the dispatch must not turn
        this file into a test that silently checks nothing."""
        self.assertIn("navigate", self.dispatched)
        self.assertGreaterEqual(len(self.dispatched), 8)

    def test_nothing_documented_is_missing_from_the_adapter(self) -> None:
        invented = self.documented - self.dispatched
        self.assertFalse(
            invented,
            f"documented but not executable: {sorted(invented)} — the model will "
            "emit these and the step will stop with 'unsupported browser action'",
        )

    def test_nothing_executable_is_left_undocumented(self) -> None:
        hidden = self.dispatched - self.documented - set(BROWSER_ACTIONS_NOT_EXECUTED)
        # `check`/`uncheck` are described inside the `click` entry rather than
        # given rows of their own; they are the same gesture.
        hidden -= {"check", "uncheck"}
        self.assertFalse(
            hidden,
            f"executable but undocumented: {sorted(hidden)} — the Configurator "
            "cannot author what it was never told exists",
        )

    def test_the_actions_that_do_not_run_are_named_as_such(self) -> None:
        for name in BROWSER_ACTIONS_NOT_EXECUTED:
            self.assertIn(name, self.dispatched)
            self.assertNotIn(name, self.documented)


class TheBlockTellsTheAuthorWhatMattersTest(unittest.TestCase):
    """The block is prompt text; these are the parts a workflow fails without."""

    def setUp(self) -> None:
        self.block = browser_action_vocabulary_block()

    def test_it_explains_how_a_runtime_value_reaches_a_later_action(self) -> None:
        self.assertIn("{{cafe_id}}", self.block)
        self.assertIn("extract", self.block)

    def test_it_says_which_assertion_proves_you_are_still_signed_in(self) -> None:
        self.assertIn("url_not_contains", self.block)

    def test_it_distinguishes_typing_from_setting_a_value(self) -> None:
        """A rich-text editor ignores a set value; the post submits empty."""
        self.assertIn("rich-text", self.block)

    def test_it_warns_that_placeholders_stop_the_step(self) -> None:
        self.assertIn("configured_", self.block)


class TheConfiguratorPromptCarriesItTest(unittest.TestCase):
    def test_the_marker_is_replaced_not_left_in_the_prompt(self) -> None:
        from agent.configurator import build_configurator_system_prompt

        prompt = build_configurator_system_prompt()
        self.assertNotIn("{{BROWSER_ACTION_VOCABULARY_BLOCK}}", prompt)
        self.assertIn("url_not_contains", prompt)
        self.assertIn("{{cafe_id}}", prompt)


if __name__ == "__main__":
    unittest.main()
