"""An edit made while drafting was thrown away, and the reply said it landed.

While a draft is being built, `_preserve_flow_for_additive_request` merges the
model's flow with the previous one whenever the message "looks additive". The
markers it looks for include `단계` — which is simply the Korean word for
"step", so nearly every request to change a step trips it.

The merge then walked the *previous* flow first and skipped any step whose id
it had already seen. A corrected step therefore arrived, matched an id, and
was discarded. The assistant's reply is written from what the model produced,
so the user was told about a fix that no longer existed in the draft.

Found on a live draft, not in theory: the authoring gate refused a save
because two condition branches compared `{{run_tests.output}}`, a reference
nothing binds. Told exactly that, the Configurator rewrote every reference and
replaced one condition step — and the draft came back unchanged except for the
one step whose id was new. Saving was refused a second time, naming the same
reference it had just been asked to fix.

The sibling case on a saved agent is covered in `test_builder_revision.py`
(`test_the_word_that_means_step_does_not_revert_the_edit`) and was fixed by
skipping the merge entirely there. That is not available here — during
drafting the merge earns its keep, because a model asked for one more step
tends to reply with only that step and lose the rest. So the merge stays, and
becomes safe under the trigger instead: the model's version wins for every
step it sent, and only steps it omitted are restored.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.configurator import (  # noqa: E402
    AgentDraft,
    _looks_like_additive_workflow_request,
    _preserve_flow_for_additive_request,
)
from agent.agent_models import WorkflowStep  # noqa: E402


def _step(step_id: str, name: str) -> WorkflowStep:
    return WorkflowStep(id=step_id, name=name, type="llm", description=name)


def _draft(*steps: WorkflowStep) -> AgentDraft:
    return AgentDraft(
        name="점검",
        description="점검",
        system_prompt="점검한다",
        provider_id="anthropic",
        flow=list(steps),
    )


# The word a user types when asking for a change, which is also the word the
# additive heuristic keys on.
ASKING_TO_FIX_A_STEP = "triage 단계의 참조가 바인딩되지 않는대. 고쳐줘."


class EditSurvivesAdditiveWordingTests(unittest.TestCase):
    def test_the_corrected_step_is_the_one_that_survives(self) -> None:
        previous = _draft(_step("triage", "분기"), _step("notify", "알림"))
        corrected = _draft(_step("triage", "분기 — 참조 수정"), _step("notify", "알림"))

        merged = _preserve_flow_for_additive_request(
            previous, corrected, user_message=ASKING_TO_FIX_A_STEP
        )

        self.assertEqual([s.id for s in merged.flow], ["triage", "notify"])
        self.assertEqual(
            merged.flow[0].name,
            "분기 — 참조 수정",
            "the model's edit must win over the version it replaced",
        )

    def test_a_step_the_model_forgot_is_still_restored(self) -> None:
        """The reason the merge exists at all — it must keep working."""

        previous = _draft(
            _step("git_status", "git"), _step("run_tests", "테스트"), _step("notify", "알림")
        )
        # The model answers with only the step it was asked about.
        partial = _draft(_step("run_tests", "테스트 — 경로 수정"))

        merged = _preserve_flow_for_additive_request(
            previous, partial, user_message=ASKING_TO_FIX_A_STEP
        )

        self.assertEqual(
            [s.id for s in merged.flow], ["git_status", "run_tests", "notify"]
        )
        self.assertEqual(merged.flow[1].name, "테스트 — 경로 수정")

    def test_a_genuinely_new_step_is_appended_not_dropped(self) -> None:
        previous = _draft(_step("triage", "분기"))
        grown = _draft(_step("triage", "분기"), _step("streak_judge", "연속 판정"))

        merged = _preserve_flow_for_additive_request(
            previous, grown, user_message="연속 판정 단계를 추가해줘."
        )

        self.assertEqual([s.id for s in merged.flow], ["triage", "streak_judge"])


class DeletionIsNotAnAdditionTests(unittest.TestCase):
    """"지워줘" must reach the merge as a deletion, not as an addition.

    The negative marker list held 삭제 and 제거 — the nouns a specification is
    written in — but not the verb a person actually types. So "이 단계 지워줘"
    read as additive, the merge restored the step the model had just removed,
    and the reply said the deletion was made.

    Measured on a live draft: asked twice to remove one leftover notify step,
    the Configurator removed it both times and the flow came back with it still
    present, at the same length, with the assistant listing the correct shorter
    workflow in prose.
    """

    def test_the_verb_people_use_reads_as_a_deletion(self) -> None:
        for message in (
            "notify_result 단계를 지우고 7단계로 맞춰줘.",
            "이 단계 지워줘.",
            "알림 단계 하나 빼줘.",
        ):
            with self.subTest(message=message):
                self.assertFalse(
                    _looks_like_additive_workflow_request(message),
                    "a deletion must not be merged as an addition",
                )

    def test_the_additive_case_is_untouched(self) -> None:
        for message in ("연속 판정 단계를 추가해줘.", "알림 기능도 넣어줘."):
            with self.subTest(message=message):
                self.assertTrue(_looks_like_additive_workflow_request(message))

    def test_a_deleted_step_stays_deleted(self) -> None:
        """End to end through the merge, which is what actually failed."""

        previous = _draft(_step("notify", "알림"), _step("notify_result", "합친 알림"))
        without_it = _draft(_step("notify", "알림"))

        merged = _preserve_flow_for_additive_request(
            previous, without_it, user_message="notify_result 단계를 지워줘."
        )

        self.assertEqual([s.id for s in merged.flow], ["notify"])


if __name__ == "__main__":
    unittest.main()
