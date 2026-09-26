"""A failed step kept its checkpoint and threw away the reason for it.

`_finish_execution` writes what an attempt produced into the step's row — for
a browser step that is the whole `browser_action` block: which action ran,
what the page was, what the error said. The routing that happens next then
merges its own key into "the step's output" and stores the result.

It merged into the **in-memory** `step` dict the loop had been carrying, not
the row that had just been written. So the store ended up with a version that
never contained the evidence, and both routes did it: `retry` left
`{"last_retry_error": {"message": "... did not complete."}}`, and the
`ask_user` park that follows a spent retry left a checkpoint and a reason.

Measured on a real run: a post editor step failed twice and parked on a
person, and the only thing recorded anywhere — row, log, notification — was
the sentence "Browser action step '제목·본문 입력' did not complete." The
observations naming the failing action had been in that column moments
earlier. Nobody could act on what was left, which is the whole point of
parking on a person.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from tests.source_lint import read_module_source  # noqa: E402


class BothRoutesReadTheRowTests(unittest.TestCase):
    """Source-shape, because the alternative is standing up a whole run.

    Both merges must start from the stored row. Checked as text so a future
    edit that reintroduces `dict(step.get("output"))` on either path fails
    here rather than in a post editor at nine in the morning.
    """

    def setUp(self) -> None:
        self.source = read_module_source("agent", "task_orchestrator")

    def test_the_retry_route_merges_into_the_stored_row(self) -> None:
        self.assertIn(
            'stored = store.get_task_step(failed_step["id"]) or failed_step',
            self.source,
            "the retry merges into a stale copy and drops the attempt's evidence",
        )
        self.assertIn(
            'output = dict(stored.get("output") or failed_step.get("output") or {})',
            self.source,
        )

    def test_the_park_route_merges_into_the_stored_row(self) -> None:
        self.assertIn(
            'stored = store.get_task_step(step["id"]) or step',
            self.source,
            "the park hands a person a checkpoint with the facts removed",
        )
        self.assertIn(
            'output = dict(stored.get("output") or step.get("output") or {})',
            self.source,
        )

    def test_neither_route_still_merges_into_the_passed_dict(self) -> None:
        """The exact expressions that caused it, gone from both places."""

        self.assertNotIn('output = dict(failed_step.get("output") or {})', self.source)
        self.assertNotIn(
            'output = dict(step.get("output") or {})\n    output["checkpoint"]',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
