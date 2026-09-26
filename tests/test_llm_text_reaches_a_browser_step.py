"""A browser step could not type what an llm step had just written.

Two things had to be true for "write today's post, then post it" to be
expressible, and neither was.

The run scope published `status` for an `llm` step and nothing else, so the
text it wrote — sitting in `output["result"]` — never became a name. And the
placeholder gate judged a browser action's references against `extract` names
alone, so even once the name existed the save was refused with "not a name an
earlier extract creates".

Measured: the agent that posts to a cafe every morning had to be built with
the article as a literal string inside the browser action, which means every
run posts the same one. The Configurator reached that conclusion itself,
removed the writing step, and said why.

Order still decides. A reference to a step that has not run yet parks exactly
like a missing extract, so facts are accumulated as the flow is walked rather
than gathered up front — which is what `_producible_names` does, deliberately,
for a different question (a condition can be reached by a backward arm).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.workflow_contract import _check_placeholder_targets  # noqa: E402

WRITE_URL = "https://cafe.naver.com/ca-fe/cafes/1/menus/8/articles/write"


def _llm(step_id: str) -> dict:
    return {
        "id": step_id,
        "type": "llm",
        "name": step_id,
        "instruction": "오늘의 글을 써라.",
    }


def _browser(step_id: str, text: str) -> dict:
    return {
        "id": step_id,
        "type": "browser_action",
        "name": step_id,
        "actions": [
            {"type": "navigate", "url": WRITE_URL},
            {"type": "fill", "selector": "textarea.textarea_input", "text": text},
        ],
    }


def _blocking(steps: list[dict]) -> list[str]:
    return [
        f.detail.get("value", "")
        for f in _check_placeholder_targets(steps)
        if f.detail.get("field") == "text"
    ]


class LlmTextIsAcceptedTests(unittest.TestCase):
    def test_a_browser_step_may_type_an_earlier_llm_steps_text(self) -> None:
        steps = [_llm("compose"), _browser("post", "{{steps.compose.text}}")]

        self.assertEqual(_blocking(steps), [])

    def test_a_step_still_cannot_use_a_later_steps_text(self) -> None:
        """Forward references park at run time, so they stay refused."""

        steps = [_browser("post", "{{steps.compose.text}}"), _llm("compose")]

        self.assertEqual(_blocking(steps), ["{{steps.compose.text}}"])

    def test_a_step_cannot_use_its_own_text(self) -> None:
        """Its own fact does not exist until after it has run."""

        steps = [_browser("post", "{{steps.post.text}}")]

        self.assertEqual(_blocking(steps), ["{{steps.post.text}}"])

    def test_a_name_nothing_publishes_is_still_refused(self) -> None:
        steps = [_llm("compose"), _browser("post", "{{steps.nobody.text}}")]

        self.assertEqual(_blocking(steps), ["{{steps.nobody.text}}"])

    def test_a_bare_placeholder_is_still_refused(self) -> None:
        """The stub shape this check was written for, unaffected."""

        steps = [_llm("compose"), _browser("post", "configured_post_title")]

        self.assertEqual(_blocking(steps), ["configured_post_title"])


if __name__ == "__main__":
    unittest.main()
