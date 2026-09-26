"""Two browser steps looked continuous and were not, and nothing said so.

`_prepare_browser_session_for_execution` creates a *new* session per step and passes
the previous one's `storage_state_path` into it. Cookies survive, so a login
survives; the page the previous step left open does not. Every browser step
begins on a blank page.

The Configurator had no way to know. It designed the obvious shape — one step
that navigates to a post editor, a second that fills the form — and the second
step waited thirty seconds for a title box on a blank page and timed out,
three runs in a row:

    Page.fill: Timeout 30000ms exceeded.
      - waiting for locator("textarea.textarea_input")

The selector was right; it had been read off the live page. The step it ran in
was on nothing.

Same rule as the rest of the authoring vocabulary: the executor is the source
of truth, so the constraint is stated where the actions are listed rather than
left to be discovered by a workflow that runs at nine in the morning.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import browser_action_vocabulary_block  # noqa: E402
from code_bridge_core.configurator import (  # noqa: E402
    build_configurator_system_prompt,
)
from tests.source_lint import read_module_source  # noqa: E402


class TheRuleIsPublishedTests(unittest.TestCase):
    def test_the_block_says_a_browser_step_starts_blank(self) -> None:
        block = browser_action_vocabulary_block()

        self.assertIn("blank page", block)
        self.assertIn("navigate", block)

    def test_it_reaches_the_configurator_prompt(self) -> None:
        self.assertIn("blank page", build_configurator_system_prompt())


class TheRuleIsTrueTests(unittest.TestCase):
    """What the block claims, checked against the code that makes it so.

    Source-shape rather than a live run: standing up two real browser sessions
    to observe that one does not inherit the other's page is a slow way to
    assert something the session builder states outright.
    """

    def test_only_the_storage_state_is_handed_forward(self) -> None:
        source = read_module_source("agent", "task_orchestrator")
        start = source.index("def _prepare_browser_session_for_execution")
        body = source[start : start + 2000]

        self.assertIn('metadata["input_storage_state_path"] = previous.get(', body)
        # If a URL ever starts carrying over, this fails and the prompt text
        # above has to be corrected rather than quietly becoming untrue.
        self.assertNotIn("previous_url", body)
        self.assertNotIn('previous.get("url")', body)


if __name__ == "__main__":
    unittest.main()
