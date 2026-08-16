"""Saving a workflow that is guaranteed to stall must not read as success.

The Configurator writes `configured_cafe_url` into a browser step's target when
it does not know the real address yet, and the runtime adapter honestly parks
such a run waiting for a human (`browser_action_adapter._requires_user_target`).
Until this gate existed, the parking was the *first* anyone heard of it: the
agent was already saved, already assigned a task, and already scheduled — so
the news arrived as a run stuck at 3am, from a commit that had returned 200.

This file pins the three answers the commit paths now give:

- a workflow that cannot run as written is **refused**, with a machine-readable
  list naming the step and the exact action index a client can put an input box
  on. Nothing is written: no agent, no task, no schedule, and the builder
  session survives so the author can fix the draft and commit again;
- `commit_incomplete=true` **saves it anyway** — a deliberately unfinished draft
  is a real thing to want — but the response says so rather than pretending the
  save was clean;
- a missing server-side browser runtime **does not block the commit** (the
  install is not the author's to perform) and is stated in
  `commit_result.readiness`, so nothing downstream can report an agent as ready
  when the runtime it needs is absent.

The refusal is a 400, matching every other "this workflow cannot be saved as
written" refusal on this router (`_normalize_agent_workflow`,
`agent/workflow_v2.py`). One class of refusal, one status code.
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
from agent.browser_action_adapter import reset_browser_readiness_cache  # noqa: E402
from agent.configurator import create_builder_session  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


UNREADY_RUNTIME = {
    "ready": False,
    "playwright_python": False,
    "chromium_executable": False,
    "install_command": "/opt/venv/bin/python -m playwright install chromium",
    "message": "Playwright is not installed in this server's environment.",
}

READY_RUNTIME = {
    "ready": True,
    "playwright_python": True,
    "chromium_executable": True,
    "install_command": "/opt/venv/bin/python -m playwright install chromium",
    "message": "",
}


def _browser_step(url: str) -> dict:
    return {
        "id": "open_cafe",
        "name": "카페 열기",
        "type": "browser_action",
        "description": "카페 글쓰기 페이지를 연다.",
        "actions": [
            {"type": "navigate", "url": url},
            {"type": "click", "selector": "#write"},
        ],
    }


def _draft(flow: list[dict]) -> dict:
    return {
        "name": "카페 글쓰기",
        "description": "카페에 글을 올린다.",
        "system_prompt": "You post to a cafe.",
        "provider_id": "openai",
        "tools": [],
        "flow": flow,
        "memory_seeds": [],
    }


CLEAN_FLOW = [
    {
        "id": "check_disk",
        "name": "Check disk",
        "description": "df -h 로 여유 공간을 읽는다.",
        "success_criteria": "여유 공간 비율을 얻었다",
    }
]

PLACEHOLDER_DRAFT = _draft([_browser_step("configured_cafe_url")])
RESOLVED_DRAFT = _draft([_browser_step("https://cafe.example.com/write")])
CLEAN_DRAFT = _draft(CLEAN_FLOW)


class CommitContractGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "commit_contract_gate.db"
        agent_store._agent_store = None
        schedule_store._store = None
        reset_browser_readiness_cache()

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        reset_browser_readiness_cache()
        self._tmp.cleanup()

    # --- helpers ---------------------------------------------------------

    def _session(self, *turns: tuple[str, str]):
        session = create_builder_session(system_prompt="test")
        for role, content in turns:
            session.messages.append({"role": role, "content": content})
        return session

    def _readiness(self, snapshot):
        """Pin the readiness answer this request will judge against.

        Both getters are patched: the sync cache is what the route reads first,
        and the async one is the fall-through for a browser flow with a cold
        cache. Leaving either real would let a probe of *this* machine decide
        the assertion.
        """
        return mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=snapshot),
            get_browser_runtime_readiness=mock.AsyncMock(return_value=snapshot),
        )

    def _commit(self, draft: dict, *, session=None, **extra):
        session = session or self._session(("user", "카페에 글 올려줘"))
        body = {"session_id": session.session_id, "draft": draft}
        body.update(extra)
        return self.client.post("/api/agent/builder/commit", json=body)

    def _agent_count(self) -> int:
        return agent_store.get_agent_store().count_agents()

    # --- refusal ---------------------------------------------------------

    def test_a_placeholder_browser_target_is_refused_with_the_unresolved_list(self):
        with self._readiness(READY_RUNTIME):
            response = self._commit(PLACEHOLDER_DRAFT)

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        self.assertEqual(payload["error"], "unresolved_browser_targets")
        self.assertEqual(len(payload["unresolved"]), 1, payload["unresolved"])

        finding = payload["unresolved"][0]
        self.assertEqual(finding["code"], "unresolved_browser_target")
        self.assertEqual(finding["severity"], "blocking")
        self.assertEqual(finding["step_id"], "open_cafe")
        detail = finding["detail"]
        self.assertEqual(detail["step_id"], "open_cafe")
        # 0-based, so a client can patch `flow[i].actions[action_index]`
        # directly instead of guessing which end the numbering started at.
        self.assertEqual(detail["action_index"], 0)
        self.assertEqual(detail["action_type"], "navigate")
        self.assertEqual(detail["field"], "url")
        self.assertEqual(detail["value"], "configured_cafe_url")
        self.assertTrue(finding["ask"].strip())

        # Every existing client renders `detail` verbatim, so it must be a
        # sentence and it must name the way out.
        self.assertIsInstance(payload["detail"], str)
        self.assertIn("commit_incomplete", payload["detail"])
        self.assertTrue(payload["can_save_incomplete"])

    def test_a_refused_commit_writes_nothing_and_keeps_the_session(self):
        session = self._session(("user", "카페에 매일 아침 9시에 글 올려줘"))

        with self._readiness(READY_RUNTIME):
            response = self._commit(PLACEHOLDER_DRAFT, session=session)

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self._agent_count(), 0)
        # The draft the author must fix is still there: a refusal that also
        # destroyed the session would make the fix impossible.
        retried = self.client.post(
            "/api/agent/builder/commit",
            json={"session_id": session.session_id, "draft": RESOLVED_DRAFT},
        )
        self.assertEqual(retried.status_code, 200, retried.text)

    def test_a_template_reference_to_a_missing_step_is_refused(self):
        draft = _draft(
            [
                {
                    "id": "summarize",
                    "name": "요약",
                    "description": "{{steps.run_flutter_test}} 결과를 요약한다.",
                }
            ]
        )

        with self._readiness(READY_RUNTIME):
            response = self._commit(draft)

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        self.assertEqual(payload["error"], "unknown_step_reference")
        self.assertEqual(payload["unresolved"], [])
        self.assertEqual(
            payload["unknown_step_references"][0]["detail"]["reference"],
            "run_flutter_test",
        )
        self.assertEqual(self._agent_count(), 0)

    # --- the escape hatch -------------------------------------------------

    def test_commit_incomplete_saves_the_draft_and_still_says_what_is_missing(self):
        with self._readiness(READY_RUNTIME):
            response = self._commit(PLACEHOLDER_DRAFT, commit_incomplete=True)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["agent"]["id"])
        self.assertEqual(self._agent_count(), 1)

        readiness = result["commit_result"]["readiness"]
        self.assertFalse(readiness["ok"])
        self.assertTrue(readiness["saved_incomplete"])
        self.assertEqual(len(readiness["unresolved_targets"]), 1)
        self.assertEqual(
            readiness["unresolved_targets"][0]["detail"]["value"],
            "configured_cafe_url",
        )
        # Saving it anyway is not the same as saving it cleanly, and the
        # sentence a user reads has to carry that.
        self.assertIn("configured_cafe_url", result["commit_result"]["summary"])

    # --- a missing runtime is reported, not refused ------------------------

    def test_a_missing_browser_runtime_commits_and_is_named_in_the_response(self):
        with self._readiness(UNREADY_RUNTIME):
            response = self._commit(RESOLVED_DRAFT)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["agent"]["id"])

        readiness = result["commit_result"]["readiness"]
        self.assertFalse(readiness["ok"])
        self.assertFalse(readiness["saved_incomplete"])
        runtime = readiness["browser_runtime"]
        self.assertIsNotNone(runtime)
        self.assertFalse(runtime["ready"])
        self.assertEqual(runtime["install_command"], UNREADY_RUNTIME["install_command"])
        self.assertEqual(runtime["step_ids"], ["open_cafe"])
        self.assertIn("open_cafe", readiness["warnings"][0]["detail"]["step_ids"])
        # The summary must not read as an unqualified success.
        self.assertIn(UNREADY_RUNTIME["install_command"], result["commit_result"]["summary"])

    def test_a_ready_browser_runtime_leaves_the_commit_clean(self):
        with self._readiness(READY_RUNTIME):
            response = self._commit(RESOLVED_DRAFT)

        self.assertEqual(response.status_code, 200, response.text)
        readiness = response.json()["commit_result"]["readiness"]
        self.assertTrue(readiness["ok"])
        self.assertIsNone(readiness["browser_runtime"])
        self.assertEqual(readiness["warnings"], [])

    def test_unknown_readiness_is_never_reported_as_ready(self):
        # A cold cache on a flow with no browser step: nothing is known, so
        # nothing is claimed — and no probe is started to find out.
        probe = mock.AsyncMock(return_value=UNREADY_RUNTIME)
        with mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=None),
            get_browser_runtime_readiness=probe,
        ):
            response = self._commit(CLEAN_DRAFT)

        self.assertEqual(response.status_code, 200, response.text)
        probe.assert_not_awaited()
        readiness = response.json()["commit_result"]["readiness"]
        self.assertTrue(readiness["ok"])
        self.assertIsNone(readiness["browser_runtime"])

    def test_a_browser_flow_with_a_cold_cache_asks_once(self):
        # For a browser flow, "unknown" is not good enough — that workflow's
        # whole fate depends on the answer — so the cache-backed getter is
        # awaited. It probes at most once per TTL, not once per request.
        probe = mock.AsyncMock(return_value=UNREADY_RUNTIME)
        with mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=None),
            get_browser_runtime_readiness=probe,
        ):
            response = self._commit(RESOLVED_DRAFT)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(probe.await_count, 1)
        runtime = response.json()["commit_result"]["readiness"]["browser_runtime"]
        self.assertFalse(runtime["ready"])

    def test_a_cached_snapshot_is_used_without_probing(self):
        probe = mock.AsyncMock(return_value=READY_RUNTIME)
        with mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=UNREADY_RUNTIME),
            get_browser_runtime_readiness=probe,
        ):
            response = self._commit(RESOLVED_DRAFT)

        self.assertEqual(response.status_code, 200, response.text)
        probe.assert_not_awaited()
        self.assertFalse(
            response.json()["commit_result"]["readiness"]["browser_runtime"]["ready"]
        )

    # --- the ordinary case is untouched -----------------------------------

    def test_a_clean_flow_commits_exactly_as_before(self):
        session = self._session(("user", "매일 아침 9시에 디스크 확인해줘"))

        with self._readiness(READY_RUNTIME):
            response = self._commit(CLEAN_DRAFT, session=session)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        outcome = result["commit_result"]
        self.assertTrue(outcome["agent"]["created"])
        self.assertTrue(outcome["task"]["created"])
        self.assertTrue(outcome["schedule"]["created"])
        self.assertTrue(outcome["runs_unattended"])
        self.assertTrue(outcome["readiness"]["ok"])
        self.assertEqual(outcome["readiness"]["message"], "")


class AgentWriteRoutesContractGateTest(unittest.TestCase):
    """The same gate on the routes that do not go through the builder.

    A workflow does not become safe by arriving through a different door.
    `POST /agents` and `PATCH /agents/{id}` are how the dashboard, a script and
    the phone write an agent definition, so a check only the builder performed
    would be a check the product does not have.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "agent_write_contract_gate.db"
        agent_store._agent_store = None
        schedule_store._store = None
        reset_browser_readiness_cache()

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        reset_browser_readiness_cache()
        self._tmp.cleanup()

    def _readiness(self, snapshot):
        return mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=snapshot),
            get_browser_runtime_readiness=mock.AsyncMock(return_value=snapshot),
        )

    def _create(self, flow: list[dict], **extra):
        body = {
            "name": "카페 글쓰기",
            "system_prompt": "You post to a cafe.",
            "flow_json": flow,
        }
        body.update(extra)
        return self.client.post("/api/agent/agents", json=body)

    def test_post_agents_refuses_a_placeholder_target(self):
        with self._readiness(READY_RUNTIME):
            response = self._create([_browser_step("configured_cafe_url")])

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        self.assertEqual(payload["error"], "unresolved_browser_targets")
        self.assertEqual(payload["unresolved"][0]["detail"]["action_index"], 0)
        self.assertEqual(agent_store.get_agent_store().count_agents(), 0)

    def test_post_agents_honours_commit_incomplete(self):
        with self._readiness(READY_RUNTIME):
            response = self._create(
                [_browser_step("configured_cafe_url")],
                commit_incomplete=True,
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["id"])
        self.assertTrue(payload["readiness"]["saved_incomplete"])
        self.assertEqual(len(payload["readiness"]["unresolved_targets"]), 1)

    def test_post_agents_reports_a_missing_runtime_without_refusing(self):
        with self._readiness(UNREADY_RUNTIME):
            response = self._create([_browser_step("https://cafe.example.com/write")])

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["readiness"]["browser_runtime"]["ready"])
        self.assertEqual(
            payload["readiness"]["browser_runtime"]["install_command"],
            UNREADY_RUNTIME["install_command"],
        )

    def test_post_agents_stays_quiet_when_there_is_nothing_to_report(self):
        with self._readiness(READY_RUNTIME):
            response = self._create(CLEAN_FLOW)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("readiness", response.json())

    def test_patch_agents_refuses_a_placeholder_target(self):
        with self._readiness(READY_RUNTIME):
            created = self._create(CLEAN_FLOW)
        agent_id = created.json()["id"]

        with self._readiness(READY_RUNTIME):
            response = self.client.patch(
                f"/api/agent/agents/{agent_id}",
                json={"flow_json": [_browser_step("configured_cafe_url")]},
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"], "unresolved_browser_targets")
        stored = agent_store.get_agent_store().get_agent(agent_id)
        self.assertEqual(stored["flow_json"][0]["id"], "check_disk")

    def test_patch_agents_honours_commit_incomplete_and_reports_it(self):
        with self._readiness(READY_RUNTIME):
            created = self._create(CLEAN_FLOW)
        agent_id = created.json()["id"]

        with self._readiness(READY_RUNTIME):
            response = self.client.patch(
                f"/api/agent/agents/{agent_id}",
                json={
                    "flow_json": [_browser_step("configured_cafe_url")],
                    "commit_incomplete": True,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["readiness"]["saved_incomplete"])
        # `commit_incomplete` is a route-level answer, never a stored column.
        self.assertNotIn("commit_incomplete", payload)
        stored = agent_store.get_agent_store().get_agent(agent_id)
        self.assertEqual(stored["flow_json"][0]["id"], "open_cafe")

    def test_patch_without_a_flow_is_not_gated(self):
        with self._readiness(UNREADY_RUNTIME):
            created = self._create([_browser_step("https://cafe.example.com/write")])
        agent_id = created.json()["id"]

        with self._readiness(UNREADY_RUNTIME):
            response = self.client.patch(
                f"/api/agent/agents/{agent_id}",
                json={"description": "이름만 바꾼다"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        # No workflow was submitted, so no workflow was judged.
        self.assertNotIn("readiness", response.json())


if __name__ == "__main__":
    unittest.main()
