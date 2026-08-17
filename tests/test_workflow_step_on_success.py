"""``on_success`` is a formal WorkflowStep field, not an ``extra="allow"`` accident.

The runtime has always honoured ``on_success`` (``workflow_v2.
normalize_success_policy``, COMMON_STEP_FIELDS), but the Configurator's
``WorkflowStep`` pydantic model only carried it because the model permits
unknown extras. Nothing pinned that: tightening the model config, or any
rebuild path that re-validates only declared fields, could have silently
dropped a success route the author wrote. This file pins the three places the
field must exist on purpose:

- the pydantic model declares it (T-B-09),
- a draft carrying it survives the builder commit serialization
  (``routes.agents.builder_commit`` line: ``normalize_workflow([step.model_dump()
  for step in draft.flow])``),
- the Configurator system prompt tells the model the field exists, next to
  ``on_failure``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.agent_models import WorkflowStep  # noqa: E402
from agent.configurator import (  # noqa: E402
    build_configurator_system_prompt,
    create_builder_session,
)
from agent.workflow_v2 import normalize_workflow  # noqa: E402


class WorkflowStepOnSuccessModelTest(unittest.TestCase):
    def test_on_success_is_a_declared_model_field_not_an_extra(self) -> None:
        self.assertIn("on_success", WorkflowStep.model_fields)

        step = WorkflowStep.model_validate(
            {
                "id": "check",
                "type": "llm",
                "name": "Check",
                "on_success": {"type": "end"},
            }
        )
        self.assertEqual(step.on_success, {"type": "end"})
        # Declared means declared: it must not be living in the extras bag.
        self.assertNotIn("on_success", step.__pydantic_extra__ or {})

    def test_on_success_defaults_to_continue_like_the_runtime(self) -> None:
        step = WorkflowStep(name="Check")
        self.assertEqual(step.on_success, "continue")


class WorkflowStepOnSuccessCommitPathTest(unittest.TestCase):
    def test_draft_on_success_survives_builder_commit_serialization(self) -> None:
        """Mirror of ``builder_commit``: model_dump each step, then normalize."""

        flow = [
            WorkflowStep(
                id="nightly_check",
                type="llm",
                name="Nightly check",
                on_success={"type": "end"},
                on_failure={"type": "goto_step", "target_step_id": "diagnose"},
            ),
            WorkflowStep(id="diagnose", type="llm", name="Diagnose"),
        ]

        normalized = normalize_workflow([step.model_dump() for step in flow])

        self.assertEqual(normalized[0]["on_success"], {"type": "end"})
        # Absent on the draft means the runtime default, not a lost key.
        self.assertEqual(normalized[1]["on_success"], {"type": "continue"})

    def test_llm_draft_block_on_success_reaches_the_session_draft(self) -> None:
        session = create_builder_session(system_prompt="test")
        session.apply_llm_response(
            """
Draft with a success route.

```draft
{
  "name": "Site monitor",
  "description": "Checks the site and stops early when it is healthy.",
  "system_prompt": "Check the site.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {
      "id": "health_check",
      "type": "llm",
      "name": "Health check",
      "instruction": "Check whether the site responds.",
      "on_success": {"type": "end"},
      "on_failure": {"type": "goto_step", "target_step_id": "diagnose"}
    },
    {
      "id": "diagnose",
      "type": "llm",
      "name": "Diagnose",
      "instruction": "Summarize what failed."
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="성공하면 바로 끝나는 상태 점검 Agent를 만들어줘.",
        )

        self.assertIsNotNone(session.current_draft)
        steps = {step.id: step for step in session.current_draft.flow}
        self.assertEqual(steps["health_check"].on_success, {"type": "end"})


class WorkflowStepOnSuccessPromptTest(unittest.TestCase):
    def test_system_prompt_schema_block_documents_on_success(self) -> None:
        prompt = build_configurator_system_prompt()
        self.assertIn("on_success", prompt)
        # Next to on_failure in the same WorkflowStep listing, with the two
        # non-default routes an author would actually reach for.
        self.assertIn('{type:"end"}', prompt)
        self.assertIn('{type:"goto_step", target_step_id:"..."} jump on success', prompt)


if __name__ == "__main__":
    unittest.main()
