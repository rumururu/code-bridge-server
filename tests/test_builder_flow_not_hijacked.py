"""A workflow the user read and approved must survive to the database.

The reported failure, end to end. A user built a plain `flutter test` agent:
two named steps, no browser anywhere near it. What got saved was a six-step
Naver-cafe scraping workflow they had never seen. Nothing in the UI said a
substitution had happened; the flow they approved was simply gone.

Four separate mechanisms had to line up to do that, and each one looked
locally reasonable:

1. the system prompt said "에러 메시지를 요약해서 보고한다", and "메시지" was a
   marker in `_looks_like_web_automation` -> the agent was classified as
   browser automation;
2. `_ensure_playwright_tool` then injected a Playwright tool whose
   `user_examples` read "네이버 카페 게시판 열기";
3. `_raw_intent_text` folded `draft.tools` back into the text it classifies,
   so on the next pass the agent's own injected tool made
   `_looks_like_naver_cafe` true -- the classifier reading its own output as
   if it were the user's request;
4. `builder_commit` re-ran that whole pass on the way to the database, and
   `_flow_needs_operational_plan` treated any Naver flow shorter than three
   steps as unfinished -- so the approved two steps were swapped for the
   `_naver_cafe_flow()` template between the screen and the save.

Step 4's machinery is now gone entirely: A6 deleted the three flow templates
and the gate that chose between them, so there is no longer any workflow for
the server to substitute in. The names above are kept because they are what
the reported failure was made of.

Each fix is guarded by its own assertion below, because any one of the four
re-opens the chain on its own. The load-bearing property is the one in the
title: converse may add hints, and commit may normalise field shapes, but
neither may replace steps a human approved.
"""

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from agent.configurator import create_builder_session  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


# The exact draft the Configurator returned in the reported session: a shell
# step and an llm step, and a system prompt containing "에러 메시지".
LLM_RESPONSE = """
flutter 테스트 에이전트 초안입니다.

```draft
{
  "name": "Flutter Test Runner",
  "description": "프로젝트의 flutter test를 실행하고 실패를 요약한다.",
  "system_prompt": "flutter test를 실행하고, 실패한 테스트의 에러 메시지를 요약해서 보고한다.",
  "provider_id": "anthropic",
  "tools": [],
  "flow": [
    {
      "id": "run_flutter_test",
      "type": "shell",
      "name": "flutter test 실행",
      "description": "프로젝트 루트에서 flutter test를 실행한다.",
      "script_id": "scr_flutter_test",
      "success_criteria": "테스트 종료 코드와 출력이 수집됨"
    },
    {
      "id": "summarize_failures",
      "type": "llm",
      "name": "실패 요약",
      "description": "실패한 테스트와 에러 메시지를 요약한다.",
      "success_criteria": "실패 목록과 원인 요약이 작성됨"
    }
  ],
  "memory_seeds": []
}
```
"""

# The user's own sentence, verbatim -- including the markdown emphasis and the
# leading conjunction that the Configurator's own reply had introduced.
USER_MESSAGE = "그리고 **매일 오전 9시에** flutter test 돌려서 실패하면 알려줘"

APPROVED_STEPS = [
    ("run_flutter_test", "flutter test 실행"),
    ("summarize_failures", "실패 요약"),
]


class BuilderFlowNotHijackedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "builder_flow_not_hijacked.db"
        agent_store._agent_store = None
        schedule_store._store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _converse(self):
        session = create_builder_session(system_prompt="test")
        session.messages.append({"role": "user", "content": USER_MESSAGE})
        session.apply_llm_response(LLM_RESPONSE, user_message=USER_MESSAGE)
        return session

    @staticmethod
    def _step_identities(flow) -> list[tuple[str, str]]:
        return [(step["id"], step["name"]) for step in flow]

    # --- the turn ---------------------------------------------------------

    def test_the_turn_leaves_the_approved_flow_and_tool_set_alone(self):
        session = self._converse()
        draft = session.current_draft

        self.assertEqual(
            [(step.id, step.name) for step in draft.flow],
            APPROVED_STEPS,
            "the Configurator turn replaced the workflow the model proposed",
        )
        # Fault 1+2: "에러 메시지" in the system prompt must not read as browser
        # automation, so no browser tool may be conjured for a shell agent.
        self.assertEqual(
            [tool.mcp_id for tool in draft.tools],
            [],
            "a non-web agent was given browser tooling it never asked for",
        )
        self.assertEqual([step.tool_hint for step in draft.flow], [None, None])

    def test_an_injected_browser_tool_could_not_describe_itself_as_naver_cafe(self):
        # Fault 2 is only dangerous because the injected tool's own examples
        # are user-visible text that later gets re-read. Even where the tool is
        # legitimately injected, its examples must stay neutral.
        from agent.configurator import _ensure_playwright_tool  # noqa: PLC0415
        from agent.agent_models import AgentDraft  # noqa: PLC0415

        tool = _ensure_playwright_tool(AgentDraft()).tools[0]
        self.assertEqual(tool.mcp_id, "playwright")
        blurb = f"{tool.user_capability} {' '.join(tool.user_examples)}"
        self.assertNotIn("네이버", blurb)
        self.assertNotIn("카페", blurb)

    def test_the_classifier_does_not_read_its_own_tool_output_as_user_intent(self):
        # Fault 3: `_raw_intent_text` is what the keyword classifiers see. A
        # tool the pass itself injected must not appear there, or the pass
        # confirms its own guess on the next call.
        from agent.configurator import _raw_intent_text  # noqa: PLC0415
        from agent.agent_models import AgentDraft, AgentToolDraft  # noqa: PLC0415
        from agent.agent_models import MCPCategory, MCPRiskTier  # noqa: PLC0415

        draft = AgentDraft(
            name="Flutter Test Runner",
            tools=[
                AgentToolDraft(
                    mcp_id="playwright",
                    tool_names=["browser_navigate"],
                    user_capability="웹사이트를 연다.",
                    user_examples=["네이버 카페 게시판 열기"],
                    category=MCPCategory.BROWSER,
                    risk_tier=MCPRiskTier.MEDIUM,
                )
            ],
        )
        text = _raw_intent_text("flutter test 돌려줘", draft)
        self.assertNotIn("네이버 카페", text)
        self.assertNotIn("playwright", text)

    def test_a_named_flow_is_never_swapped_for_a_template(self):
        # Fault 4, isolated. A1 narrowed the swap to empty/placeholder flows;
        # A6 removed the templates outright, so `_flow_needs_operational_plan`
        # no longer exists to be asked. The property is now stronger and is
        # asserted against the pass itself: feed `enrich_draft_from_user_intent`
        # the most Naver-cafe wording there is and the flow comes back exactly
        # as it went in -- including when it is empty, which is the case the
        # templates used to fill in.
        from agent.configurator import enrich_draft_from_user_intent  # noqa: PLC0415

        session = self._converse()
        approved = session.current_draft
        naver_message = (
            "네이버 카페 게시판에서 품앗이 글을 찾아 신청하고 쪽지도 보내줘"
        )

        enriched, _ = enrich_draft_from_user_intent(
            approved,
            previous_draft=approved,
            task_draft=None,
            user_message=naver_message,
        )
        self.assertEqual(
            [(step.id, step.type, step.name) for step in enriched.flow],
            [(step.id, step.type, step.name) for step in approved.flow],
        )

        emptied = approved.model_copy(update={"flow": []})
        enriched_empty, _ = enrich_draft_from_user_intent(
            emptied,
            previous_draft=emptied,
            task_draft=None,
            user_message=naver_message,
        )
        self.assertEqual(
            enriched_empty.flow,
            [],
            "an empty flow must stay empty -- the server does not write the workflow",
        )

    # --- the commit -------------------------------------------------------

    def test_the_commit_saves_the_flow_the_user_approved(self):
        session = self._converse()
        draft_payload = session.current_draft.model_dump(mode="json")

        response = self.client.post(
            "/api/agent/builder/commit",
            json={"session_id": session.session_id, "draft": draft_payload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()

        saved = result["agent"]
        self.assertEqual(
            self._step_identities(saved["flow_json"]),
            APPROVED_STEPS,
            "commit replaced the approved workflow on the way to the database",
        )
        self.assertEqual(
            [step["type"] for step in saved["flow_json"]],
            ["shell", "llm"],
            "commit rewrote the step types the user approved",
        )
        self.assertEqual(
            [tool["mcp_id"] for tool in saved["tools_json"]],
            [],
            "commit attached tooling the user never approved",
        )
        # The commit must not have edited the draft it was handed either --
        # what came back is what was sent, field for field.
        self.assertEqual(saved["name"], draft_payload["name"])
        self.assertEqual(saved["system_prompt"], draft_payload["system_prompt"])
        self.assertEqual(saved["provider_id"], draft_payload["provider_id"])

    def test_the_stated_schedule_still_reaches_the_commit(self):
        # Removing the commit-time enrichment must not cost the user their
        # schedule: it is derived from the conversation, which is the only
        # place "매일 오전 9시" was ever said.
        session = self._converse()

        response = self.client.post(
            "/api/agent/builder/commit",
            json={
                "session_id": session.session_id,
                "draft": session.current_draft.model_dump(mode="json"),
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()

        task_id = result["task"]["id"]
        schedules = schedule_store.get_schedule_store().list_for_task(task_id)
        self.assertEqual(len(schedules), 1, schedules)
        self.assertEqual(schedules[0]["expression"], {"kind": "daily_at", "time": "09:00"})
        self.assertTrue(result["commit_result"]["runs_unattended"])

    def test_a_draft_missing_a_required_field_is_refused_not_back_filled(self):
        # The commit path used to run an enrichment pass that quietly invented
        # a name for a nameless draft. Honest refusal is the contract now.
        session = self._converse()
        draft_payload = session.current_draft.model_dump(mode="json")
        draft_payload["name"] = None

        response = self.client.post(
            "/api/agent/builder/commit",
            json={"session_id": session.session_id, "draft": draft_payload},
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("name", response.json()["detail"])
        # And nothing was written on the way to that refusal.
        listed = self.client.get("/api/agent/agents")
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(listed.json()["agents"], [])


if __name__ == "__main__":
    unittest.main()
