"""Device steps must publish what they can actually do.

The browser side learned this the hard way: with no vocabulary in front of it,
the Configurator wrote every browser step as a placeholder `navigate` plus a
screenshot, because it had no way to know `extract` or `assert` existed. The
device side had *no* published vocabulary at all — seventeen action types the
executor dispatches, and nothing telling an author they were there.

Publishing creates the opposite hazard: a table that drifts from the dispatch.
A documented action that does not exist is worse than an undocumented one,
because a model will confidently emit it and the step will stop with an
unsupported-action error. These tests read the executor's own source and fail
when the two disagree, the same guard `test_browser_action_vocabulary.py`
applies to the browser table.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.app_action_executor import (  # noqa: E402
    APP_ACTION_VOCABULARY,
    app_action_vocabulary_block,
)

EXECUTOR_SOURCE = (SERVER_DIR / "agent" / "app_action_executor.py").read_text(
    encoding="utf-8"
)


def _dispatched_action_types() -> set[str]:
    """Every action type the executor actually branches on."""
    found: set[str] = set()
    for match in re.finditer(r"action_type (?:==|in) (.+)", EXECUTOR_SOURCE):
        found.update(re.findall(r'"([a-z_]+)"', match.group(1)))
    return found


class VocabularyMatchesTheExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documented = {action.type for action in APP_ACTION_VOCABULARY}
        self.dispatched = _dispatched_action_types()

    def test_the_dispatch_was_found(self) -> None:
        """Guard the guard: a refactor that moves the dispatch must not turn
        this file into a test that silently checks nothing."""
        self.assertIn("install_app", self.dispatched)
        self.assertGreaterEqual(len(self.dispatched), 10)

    def test_nothing_documented_is_missing_from_the_executor(self) -> None:
        invented = self.documented - self.dispatched
        self.assertFalse(
            invented,
            f"documented but not executable: {sorted(invented)} — an author "
            "will write these and the step will stop on an unsupported action",
        )

    def test_nothing_executable_is_left_undocumented(self) -> None:
        hidden = self.dispatched - self.documented
        self.assertFalse(
            hidden,
            f"executable but undocumented: {sorted(hidden)} — the Configurator "
            "cannot author what it was never told exists, which is how every "
            "browser step became a placeholder",
        )


class TheBlockReachesTheAuthorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.block = app_action_vocabulary_block()

    def test_every_action_is_named_in_the_block(self) -> None:
        for action in APP_ACTION_VOCABULARY:
            with self.subTest(action=action.type):
                self.assertIn(action.type, self.block)

    def test_the_configurator_prompt_carries_it(self) -> None:
        """A vocabulary nothing injects is a vocabulary nobody reads."""
        from code_bridge_core.configurator import build_configurator_system_prompt

        prompt = build_configurator_system_prompt()
        self.assertNotIn("{{APP_ACTION_VOCABULARY_BLOCK}}", prompt)
        self.assertIn("install_app", prompt)
        self.assertIn("verify_launch", prompt)

    def test_the_browser_vocabulary_still_reaches_it_too(self) -> None:
        """Adding a second block must not displace the first."""
        from code_bridge_core.configurator import build_configurator_system_prompt

        prompt = build_configurator_system_prompt()
        self.assertNotIn("{{BROWSER_ACTION_VOCABULARY_BLOCK}}", prompt)
        self.assertIn("url_not_contains", prompt)


if __name__ == "__main__":
    unittest.main()
