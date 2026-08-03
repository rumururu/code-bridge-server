"""A commit must create what the conversation promised, and say what it did.

A first-time user told the Configurator "check my Mac's disk daily at 9am,
mail me if it drops under 15%". The Configurator answered "매일 아침 09:00
자동 실행" and the commit returned 200. The database got an agent and nothing
else: no task, no schedule. The user was told their agent runs every morning;
it never ran at all, and nothing anywhere said so.

Two separate faults produced that, and this file guards both:

- `builder_commit` only created a task (and, inside that branch, a schedule)
  when a `task_draft` object happened to reach it. The dashboard commits with
  `{session_id, draft}` and no task draft, so everything depended on the LLM
  emitting a `task_draft` block — an unenforceable hope. When it did not, the
  stated schedule was simply dropped.
- Whatever it skipped, the response looked identical: `{"agent": ...}`. A
  caller could not distinguish "scheduled, will fire at 09:00" from "created
  a name and nothing else", so every surface downstream reported success.

If this regresses, an agent that the user was told is scheduled goes back to
being an agent that never runs, silently. The response contract these tests
pin (`commit_result.agent/task/schedule`, each created-with-id or not-with-a-
reason) is the part that must not quietly shrink: reporting a schedule that
does not exist is the defect, not the missing schedule alone.

The other half of the rule is that commit must not invent work. A commit with
no timing intent still creates only an agent — and says plainly that nothing
will run on its own, rather than staying quiet and letting the user assume.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


DRAFT = {
    "name": "디스크 감시",
    "description": "맥 디스크 여유 공간을 확인하고 15% 미만이면 메일로 알린다.",
    "system_prompt": "You watch disk space.",
    "provider_id": "openai",
    "tools": [],
    "flow": [
        {
            "name": "Check disk",
            "description": "df -h 로 여유 공간을 읽는다.",
            "success_criteria": "여유 공간 비율을 얻었다",
        }
    ],
    "memory_seeds": [],
}


class BuilderCommitTruthTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "builder_commit_truth.db"
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

    def _session(self, *turns: tuple[str, str]):
        session = create_builder_session(system_prompt="test")
        for role, content in turns:
            session.messages.append({"role": role, "content": content})
        return session

    def _commit(self, session, **body_extra):
        body = {"session_id": session.session_id, "draft": DRAFT}
        body.update(body_extra)
        response = self.client.post("/api/agent/builder/commit", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _schedules_for(self, task_id: str):
        return schedule_store.get_schedule_store().list_for_task(task_id)

    # --- the walkthrough -------------------------------------------------

    def test_a_stated_daily_schedule_commits_a_task_and_an_enabled_schedule(self):
        session = self._session(
            (
                "user",
                "매일 아침 9시에 맥 디스크 남은 용량 확인해서 15% 미만이면 메일 보내줘",
            ),
            (
                "assistant",
                "디스크 감시 에이전트를 준비했습니다. 매일 아침 09:00 자동 실행합니다.",
            ),
        )

        result = self._commit(session)

        self.assertTrue(result["agent"]["id"])
        self.assertIn("task", result)
        task_id = result["task"]["id"]

        schedules = self._schedules_for(task_id)
        self.assertEqual(len(schedules), 1, schedules)
        self.assertEqual(schedules[0]["expression"], {"kind": "daily_at", "time": "09:00"})
        self.assertTrue(schedules[0]["enabled"])

        outcome = result["commit_result"]
        self.assertTrue(outcome["agent"]["created"])
        self.assertEqual(outcome["agent"]["id"], result["agent"]["id"])
        self.assertTrue(outcome["task"]["created"])
        self.assertEqual(outcome["task"]["id"], task_id)
        self.assertTrue(outcome["schedule"]["created"])
        self.assertEqual(outcome["schedule"]["id"], schedules[0]["id"])
        self.assertEqual(
            outcome["schedule"]["expression"], {"kind": "daily_at", "time": "09:00"}
        )
        self.assertTrue(outcome["runs_unattended"])
        # The agent the caller gets back must already know when it fires, so a
        # UI that only reads `agent` cannot show "never".
        self.assertIsNotNone(result["agent"]["next_fire_at"])

    def test_an_assistant_only_promise_is_still_a_promise(self):
        # The walkthrough's exact shape: the user described the job in prose
        # and the Configurator was the one that named the time. Committing
        # must honour what the user was told, not only what they typed.
        session = self._session(
            ("user", "맥 디스크 용량 좀 봐주는 에이전트 만들어줘"),
            ("assistant", "매일 아침 09:00 에 자동 실행하도록 준비했습니다."),
        )

        result = self._commit(session)

        schedules = self._schedules_for(result["task"]["id"])
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0]["expression"], {"kind": "daily_at", "time": "09:00"})
        self.assertEqual(
            result["commit_result"]["task"]["origin"],
            "synthesized_from_schedule_intent",
        )

    def test_the_original_english_sentence_schedules_the_agent(self):
        # Verbatim from the walkthrough that opened this initiative.
        session = self._session(
            ("user", "check my Mac's disk daily at 9am, mail me if under 15%"),
        )

        result = self._commit(session)

        schedules = self._schedules_for(result["task"]["id"])
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0]["expression"], {"kind": "daily_at", "time": "09:00"})
        self.assertTrue(result["commit_result"]["runs_unattended"])

    def test_a_later_manual_decision_overrides_an_earlier_schedule(self):
        # Changing your mind must not leave a schedule behind: the newest
        # statement about timing is the one that counts.
        session = self._session(
            ("user", "매일 아침 9시에 디스크 확인해줘"),
            ("assistant", "매일 09:00 자동 실행으로 준비했습니다."),
            ("user", "아니야, 수동으로 내가 부를 때만 실행할게"),
        )

        result = self._commit(session)

        self.assertNotIn("task", result)
        self.assertNotIn("schedule", result)
        outcome = result["commit_result"]
        self.assertFalse(outcome["task"]["created"])
        self.assertFalse(outcome["schedule"]["created"])
        self.assertFalse(outcome["runs_unattended"])

    # --- what it could not build, it names --------------------------------

    def test_unparsable_schedule_text_is_named_and_no_schedule_is_claimed(self):
        session = self._session(("user", "매주 월요일 오전 9시에 리포트 보내줘"))

        result = self._commit(
            session,
            task_draft={"goal": "주간 리포트 발송", "schedule": "매주 월요일 오전 9시"},
        )

        # The agent and the task are still worth keeping.
        self.assertTrue(result["agent"]["id"])
        task_id = result["task"]["id"]
        self.assertEqual(self._schedules_for(task_id), [])
        self.assertNotIn("schedule", result)

        schedule = result["commit_result"]["schedule"]
        self.assertFalse(schedule["created"])
        self.assertEqual(schedule["reason"], "unparsed_schedule_text")
        self.assertEqual(schedule["requested_text"], "매주 월요일 오전 9시")
        # The user has to be able to see their own words in the answer.
        self.assertIn("매주 월요일 오전 9시", schedule["message"])
        self.assertFalse(result["commit_result"]["runs_unattended"])
        self.assertIn("매주 월요일 오전 9시", result["commit_result"]["summary"])

    def test_a_schedule_the_store_rejects_is_reported_not_swallowed(self):
        session = self._session(("user", "매일 아침 9시에 디스크 확인해줘"))

        broken = mock.Mock()
        broken.create.side_effect = ValueError("interval.seconds must be an integer >= 60")
        with mock.patch.object(agents, "get_schedule_store", return_value=broken):
            result = self._commit(session)

        task_id = result["task"]["id"]
        self.assertEqual(self._schedules_for(task_id), [])
        schedule = result["commit_result"]["schedule"]
        self.assertFalse(schedule["created"])
        self.assertEqual(schedule["reason"], "rejected_by_schedule_store")
        self.assertIn("interval.seconds", schedule["detail"])
        self.assertFalse(result["commit_result"]["runs_unattended"])

    # --- nothing to schedule is a legitimate answer -----------------------

    def test_a_commit_with_no_timing_intent_says_nothing_will_run(self):
        session = self._session(
            ("user", "내가 물어보면 디스크 상태 알려주는 에이전트 만들어줘"),
            ("assistant", "요청하실 때 실행하는 에이전트로 준비했습니다."),
        )

        result = self._commit(session)

        self.assertTrue(result["agent"]["id"])
        self.assertNotIn("task", result)
        outcome = result["commit_result"]
        self.assertTrue(outcome["agent"]["created"])
        self.assertFalse(outcome["task"]["created"])
        self.assertEqual(outcome["task"]["reason"], "no_task_intent")
        self.assertFalse(outcome["schedule"]["created"])
        self.assertEqual(outcome["schedule"]["reason"], "no_task")
        self.assertFalse(outcome["runs_unattended"])
        self.assertIn("스스로", outcome["summary"])

    def test_a_task_without_a_schedule_is_created_and_reported_as_manual(self):
        session = self._session(("user", "리포트 정리 작업 하나 만들어줘"))

        result = self._commit(session, task_draft={"goal": "리포트 정리"})

        task_id = result["task"]["id"]
        self.assertEqual(self._schedules_for(task_id), [])
        outcome = result["commit_result"]
        self.assertTrue(outcome["task"]["created"])
        self.assertEqual(outcome["task"]["origin"], "task_draft")
        self.assertFalse(outcome["schedule"]["created"])
        self.assertEqual(outcome["schedule"]["reason"], "no_schedule_requested")
        self.assertFalse(outcome["runs_unattended"])


if __name__ == "__main__":
    unittest.main()
