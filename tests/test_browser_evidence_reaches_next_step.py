"""What a browser step read must reach the step that reasons about it.

Measured on a real run: a flow browsed a cafe board, extracted the page text,
and asked an LLM step to summarise the posts. The evidence summary handed to
that step carried the visit — status, URLs, titles, screenshot paths — and not
one character of what the step had read. The model noticed ("이 스텝에 전달된
evidence 요약에는 extract 본문이 빠져 있어") and went and read the run's row out
of SQLite itself.

It produced the right answer, which is the dangerous part: the step looked
successful, and the workaround only exists because that provider happens to
have file access. The data path was simply missing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.task_orchestrator import _workflow_step_output_summary  # noqa: E402


def _browser_output(extracted, **extra):
    return {"browser_action": {"status": "completed", "extracted": extracted, **extra}}


class ExtractedTextReachesTheNextStepTest(unittest.TestCase):
    def test_page_text_is_in_the_summary(self) -> None:
        lines = _workflow_step_output_summary(
            _browser_output([{"selector": "body", "text": "품앗이 구합니다 / 테스터 모집"}])
        )
        joined = "\n".join(lines)
        self.assertIn("품앗이 구합니다", joined)

    def test_a_named_value_is_carried_in_full(self) -> None:
        """A later step referring to the value needs the value, not a prefix."""
        lines = _workflow_step_output_summary(
            _browser_output([{"name": "cafe_id", "value": "31245773"}])
        )
        self.assertIn("extracted.cafe_id: 31245773", "\n".join(lines))

    def test_a_long_page_is_truncated_rather_than_dropped(self) -> None:
        """The alternative to a bounded excerpt was no text at all."""
        lines = _workflow_step_output_summary(
            _browser_output([{"selector": "body", "text": "x" * 20_000}])
        )
        joined = "\n".join(lines)
        self.assertIn("extracted[body]", joined)
        self.assertLess(len(joined), 6_000)

    def test_truncation_upstream_is_declared(self) -> None:
        """If `extract` already cut the page, the next step should know its
        input is partial rather than treat it as the whole page."""
        lines = _workflow_step_output_summary(
            _browser_output(
                [{"selector": "body", "text": "half a page", "truncated": True}]
            )
        )
        self.assertIn("(truncated)", "\n".join(lines))

    def test_the_visit_is_still_described(self) -> None:
        """Adding the result must not cost the context around it."""
        lines = _workflow_step_output_summary(
            _browser_output(
                [{"name": "cafe_id", "value": "1"}],
                observations=[{"action_index": 1, "url": "https://x.test/board"}],
                screenshots=["/tmp/shot.png"],
            )
        )
        joined = "\n".join(lines)
        self.assertIn("browser_action.status: completed", joined)
        self.assertIn("https://x.test/board", joined)
        self.assertIn("/tmp/shot.png", joined)

    def test_a_step_that_extracted_nothing_adds_no_noise(self) -> None:
        lines = _workflow_step_output_summary(_browser_output([]))
        self.assertFalse([line for line in lines if line.startswith("extracted")])

    def test_malformed_entries_are_skipped_not_raised_on(self) -> None:
        lines = _workflow_step_output_summary(
            _browser_output(["not a dict", {"selector": "body"}, None])
        )
        self.assertFalse([line for line in lines if line.startswith("extracted")])


if __name__ == "__main__":
    unittest.main()
