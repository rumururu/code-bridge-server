"""A narrow request came back as a smaller workflow, and the reply was silent.

The model rewrites the whole draft rather than editing it, so asking for one
thing routinely returns a workflow missing others. What is lost is invisible:
the reply describes the change that was asked for, and the only tell is the
number of step cards further down the page.

Twice in one session, on a real draft:

* asked to correct three instruction texts, it returned 7 steps for 9 — and
  listed only the instruction changes;
* asked to attach a daily schedule, it returned 2 steps for 4, dropping both
  browser steps and with them three selectors that had been read off the live
  Naver page (`textarea.textarea_input`, `.se-text-paragraph`,
  `a.BaseButton--skinGreen`). Nothing in the reply mentioned it, and the draft
  was saved before anyone counted.

`_preserve_flow_for_additive_request` does not cover this and cannot: it runs
only when the message *looks* additive, and "일정을 붙여줘" does not. Restoring
silently would also undo deliberate deletions, which this module already
carries a fix for.

So the turn is made unable to be quiet about it. Nothing is restored; the
disclosure is deterministic, written next to the gate disclosure that exists
for the same reason, and the draft is still on screen to be rejected before it
is saved.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.agent_models import WorkflowStep  # noqa: E402
from code_bridge_core.configurator import _dropped_step_disclosure  # noqa: E402


def _step(step_id: str, name: str = "", step_type: str = "llm") -> WorkflowStep:
    return WorkflowStep(
        id=step_id, name=name or step_id, type=step_type, description=name or step_id
    )


class DroppedStepDisclosureTests(unittest.TestCase):
    def test_the_case_that_was_saved_before_anyone_noticed(self) -> None:
        before = [
            _step("open_write", "글쓰기 화면 열기", "browser_action"),
            _step("compose_and_fill", "글 작성 + 입력"),
            _step("submit_post", "등록", "browser_action"),
            _step("report", "결과 알림", "notify"),
        ]
        after = [_step("write_post", "오늘의 글 작성"), _step("report_result", "알림", "notify")]

        disclosure = _dropped_step_disclosure(before_flow=before, after_flow=after)

        self.assertIsNotNone(disclosure)
        for lost in ("open_write", "compose_and_fill", "submit_post", "report"):
            self.assertIn(lost, disclosure, f"{lost} vanished without being named")
        self.assertIn("4", disclosure)
        self.assertIn("2", disclosure)

    def test_an_edit_that_keeps_every_step_says_nothing(self) -> None:
        """Rewriting a step's contents is not losing it."""

        before = [_step("triage", "분기"), _step("notify", "알림", "notify")]
        after = [_step("triage", "분기 — 참조 수정"), _step("notify", "알림", "notify")]

        self.assertIsNone(_dropped_step_disclosure(before_flow=before, after_flow=after))

    def test_growing_the_workflow_says_nothing(self) -> None:
        before = [_step("triage", "분기")]
        after = [_step("triage", "분기"), _step("streak", "연속 판정")]

        self.assertIsNone(_dropped_step_disclosure(before_flow=before, after_flow=after))

    def test_the_first_draft_of_a_conversation_says_nothing(self) -> None:
        """There is nothing to lose before there is a draft."""

        self.assertIsNone(
            _dropped_step_disclosure(before_flow=[], after_flow=[_step("a")])
        )

    def test_a_deliberate_deletion_is_still_disclosed(self) -> None:
        """Deliberately or not, the user is told — nothing is undone.

        Restoring would be the wrong cure: it would resurrect steps the user
        asked to remove. Saying what happened is right either way, and reads
        as confirmation when the removal was wanted.
        """

        before = [_step("a"), _step("ghost", "유령 단계")]
        after = [_step("a")]

        disclosure = _dropped_step_disclosure(before_flow=before, after_flow=after)

        self.assertIsNotNone(disclosure)
        self.assertIn("ghost", disclosure)


if __name__ == "__main__":
    unittest.main()
