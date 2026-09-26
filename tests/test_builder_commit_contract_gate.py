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
`code_bridge_core/workflow_v2.py`). One class of refusal, one status code.
"""

import stat
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
from agent import script_store as script_store_module  # noqa: E402
from agent.browser_action_adapter import reset_browser_readiness_cache  # noqa: E402
from code_bridge_core.configurator import create_builder_session  # noqa: E402
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

def _app_step(actions: list[dict]) -> dict:
    return {
        "id": "launch_app",
        "name": "앱 실행",
        "type": "app_action",
        "description": "기기에서 앱을 실행한다.",
        "actions": actions,
    }


# What `configurator._verify_launch_actions` writes when it could not extract a
# package name. The adapter parks on it every single time.
APP_PLACEHOLDER_DRAFT = _draft(
    [
        _app_step(
            [
                {"type": "verify_launch", "app": "installed_app_from_previous_step"},
                {"type": "wait", "seconds": 1},
                {"type": "screenshot", "label": "app_launch_result"},
            ]
        )
    ]
)

APP_RESOLVED_DRAFT = _draft(
    [
        _app_step(
            [
                {"type": "verify_launch", "package": "com.example.app"},
                {"type": "wait", "seconds": 1},
                {"type": "screenshot", "label": "app_launch_result"},
            ]
        )
    ]
)

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

    def test_an_app_action_the_device_cannot_run_is_refused(self):
        # The gate used to look only at browser steps, so this draft — written
        # by the server's own normalizer — committed as a clean 200 and then
        # parked on its first fire.
        with self._readiness(READY_RUNTIME):
            response = self._commit(APP_PLACEHOLDER_DRAFT)

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        blocking = payload["blocking"]
        self.assertEqual(len(blocking), 1, blocking)
        finding = blocking[0]
        self.assertEqual(finding["code"], "unresolved_app_target")
        self.assertEqual(finding["step_id"], "launch_app")
        self.assertEqual(finding["detail"]["action_index"], 0)
        self.assertEqual(finding["detail"]["field"], "app")
        # The `wait_reason` a parked run would have carried, so the refusal and
        # the stall name the same problem.
        self.assertEqual(finding["detail"]["reason"], "app_action_needs_package_name")
        self.assertIn(finding["ask"], payload["detail"])
        self.assertTrue(payload["can_save_incomplete"])
        self.assertEqual(self._agent_count(), 0)

    def test_a_literal_app_action_still_commits(self):
        # A real package, and evidence actions whose `label`/`seconds` the
        # adapter never treats as a target: nothing here stalls, so nothing
        # here may be refused.
        with self._readiness(READY_RUNTIME):
            response = self._commit(APP_RESOLVED_DRAFT)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self._agent_count(), 1)
        self.assertTrue(response.json()["commit_result"]["readiness"]["ok"])

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


# The script the Configurator proposed for "매일 밤 스크립트를 실행하고 종료코드가
# 0이 아니면 AI가 진단해서 알림", with the declaration block the script writer is
# now told to emit. Its `usage()` and its `@param` lines say the same thing;
# only one of them is readable by anything but a person.
PROPOSED_SCRIPT = """#!/bin/bash
# @param CHECK_DIR required 여유 공간을 확인할 디렉터리
# @param HEALTH_URL required 확인할 헬스 엔드포인트
set -euo pipefail
usage() { echo "usage: $0 CHECK_DIR HEALTH_URL" >&2; exit 2; }
[ $# -ge 2 ] || usage
df -h "$1"
curl -fsS "$2" >/dev/null
"""


class ShellScriptParameterGateTest(unittest.TestCase):
    """The defect, reproduced through the door the user came in.

    Built through the builder: the Configurator proposed a script, it was
    approved through the product's own proposal path, the agent was created
    with a daily schedule and fired — and it parked on step 1 with
    `waiting_for_user`, because the script's two required positional arguments
    were declared in its `usage()` and the shell step was bound to it with
    `script_args` empty. The runtime was right to stop. What was missing is
    that the registry threw the requirement away at registration, so no gate
    could see a stall that was certain from the moment it was saved.

    The refusal is not a dead end: `builder_provider.resolveRefusalInConversation`
    hands it back to the Configurator as the next turn, which is why the `ask`
    has to name the script and each parameter.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "script_parameter_gate.db"
        agent_store._agent_store = None
        schedule_store._store = None
        script_store_module._script_store = None
        database.init_db()
        reset_browser_readiness_cache()

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        script_store_module._script_store = None
        database.DB_PATH = self._original_db_path
        reset_browser_readiness_cache()
        self._tmp.cleanup()

    # --- helpers ---------------------------------------------------------

    def _register(self, body: str, *, name: str, filename: str) -> str:
        path = self.dir / filename
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        script = script_store_module.get_script_store().register(
            name=name, path=str(path)
        )
        return script["id"]

    def _register_legacy(self, body: str, *, name: str, filename: str) -> str:
        """A row as it exists in every database that predates this column."""
        script_id = self._register(body, name=name, filename=filename)
        with database.get_db_connection() as conn:
            conn.execute(
                "UPDATE agent_scripts SET parameters_json = NULL WHERE id = ?",
                (script_id,),
            )
            conn.commit()
        return script_id

    def _flow(self, script_id: str, args: list[str] | None = None) -> list[dict]:
        step: dict = {
            "id": "run_nightly_script",
            "type": "shell",
            "name": "야간 스크립트 실행",
            "description": "밤마다 스크립트를 돌린다.",
            "script_id": script_id,
        }
        if args is not None:
            step["script_args"] = args
        return [
            step,
            {
                "id": "diagnose",
                "type": "llm",
                "name": "진단",
                "description": "종료코드가 0이 아니면 원인을 찾는다.",
            },
        ]

    def _commit(self, flow: list[dict], **extra):
        session = create_builder_session(system_prompt="test")
        session.messages.append(
            {"role": "user", "content": "매일 밤 스크립트를 실행하고 실패하면 알려줘"}
        )
        body = {"session_id": session.session_id, "draft": _draft(flow)}
        body.update(extra)
        with mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=READY_RUNTIME),
            get_browser_runtime_readiness=mock.AsyncMock(return_value=READY_RUNTIME),
        ):
            return self.client.post("/api/agent/builder/commit", json=body)

    # --- the refusal ------------------------------------------------------

    def test_a_step_that_passes_nothing_to_a_script_that_needs_two_is_refused(self):
        script_id = self._register(PROPOSED_SCRIPT, name="야간 점검", filename="nightly.sh")

        response = self._commit(self._flow(script_id))

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        # A single class of "cannot be saved as written", one status code, one
        # error string the existing clients already recognise — without which
        # the refusal would not parse and could not be handed back to the
        # conversation.
        self.assertEqual(payload["error"], "workflow_contract_blocked")
        self.assertTrue(payload["can_save_incomplete"])

        blocking = payload["blocking"]
        self.assertEqual(len(blocking), 1, blocking)
        finding = blocking[0]
        self.assertEqual(finding["code"], "script_parameters_unfilled")
        self.assertEqual(finding["step_id"], "run_nightly_script")
        detail = finding["detail"]
        self.assertEqual(detail["script_id"], script_id)
        self.assertEqual(detail["script_name"], "야간 점검")
        self.assertEqual(detail["field"], "script_args")
        self.assertEqual(detail["parameter_names"], ["CHECK_DIR", "HEALTH_URL"])
        self.assertEqual(
            [entry["description"] for entry in detail["missing_parameters"]],
            ["여유 공간을 확인할 디렉터리", "확인할 헬스 엔드포인트"],
        )

        # Named in the sentence, because that sentence is what the user reads
        # and what the Configurator is asked to turn into questions.
        self.assertIn("야간 점검", finding["ask"])
        self.assertIn("CHECK_DIR", finding["ask"])
        self.assertIn("HEALTH_URL", finding["ask"])
        self.assertIn(finding["ask"], payload["detail"])

        self.assertEqual(agent_store.get_agent_store().count_agents(), 0)

    def test_the_same_step_with_its_arguments_filled_commits(self):
        script_id = self._register(PROPOSED_SCRIPT, name="야간 점검", filename="nightly.sh")

        response = self._commit(
            self._flow(script_id, ["/Users/me/project", "https://example.com/health"])
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(agent_store.get_agent_store().count_agents(), 1)
        readiness = response.json()["commit_result"]["readiness"]
        self.assertTrue(readiness["ok"])
        stored = agent_store.get_agent_store().get_agent(response.json()["agent"]["id"])
        self.assertEqual(
            stored["flow_json"][0]["script_args"],
            ["/Users/me/project", "https://example.com/health"],
        )

    def test_a_script_registered_before_the_interface_existed_still_commits(self):
        # The migration case. That row's requirement is unknown, and unknown
        # is not grounds to refuse someone's work — if it were, this change
        # would break every workflow saved before today.
        script_id = self._register_legacy(
            PROPOSED_SCRIPT, name="구버전 스크립트", filename="legacy.sh"
        )
        self.assertIsNone(
            script_store_module.get_script_store().get(script_id)["parameters"]
        )

        response = self._commit(self._flow(script_id))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(agent_store.get_agent_store().count_agents(), 1)
        self.assertTrue(response.json()["commit_result"]["readiness"]["ok"])

    def test_the_escape_hatch_still_saves_it_and_says_what_is_missing(self):
        script_id = self._register(PROPOSED_SCRIPT, name="야간 점검", filename="nightly.sh")

        response = self._commit(self._flow(script_id), commit_incomplete=True)

        self.assertEqual(response.status_code, 200, response.text)
        readiness = response.json()["commit_result"]["readiness"]
        self.assertFalse(readiness["ok"])
        self.assertTrue(readiness["saved_incomplete"])
        self.assertIn("CHECK_DIR", readiness["message"])


DISK_WATCH_FLOW = [
    {
        "id": "usage_gate",
        "type": "condition",
        "name": "사용률 90% 초과 판정",
        "description": "루트 디스크 사용률이 90%를 넘었는지 비교한다.",
        "branches": [
            {
                "label": "90% 초과",
                "when": {"left": "{{disk_check.USED_PCT}}", "op": "gt", "right": "90"},
                "target_step_id": "diagnose",
            },
            {"label": "정상", "when": None, "target_step_id": "healthy_note"},
        ],
    },
    {
        "id": "diagnose",
        "type": "llm",
        "name": "AI 디스크 진단",
        "description": "원인을 진단한다.",
        "observation": "{{disk_check}} 단계의 표준 출력과 종료 코드",
    },
    {
        "id": "alert",
        "type": "notify",
        "name": "경고 알림 전송",
        "description": "진단 결과를 보낸다.",
        "on_success": {"type": "end"},
        "notify": {"title": "경고", "body": "확인 필요", "level": "warning"},
    },
    {
        "id": "healthy_note",
        "type": "notify",
        "name": "정상 기록",
        "description": "한 줄만 남긴다.",
        "on_success": {"type": "end"},
        "notify": {"title": "정상", "body": "임계치 아래", "level": "info"},
    },
]


class DoomedPredicateGateTest(unittest.TestCase):
    """The reported commit, refused through the door it came in.

    Built through the dashboard: the Configurator proposed a disk-usage
    script, it was approved and registered with its parameters, and the agent
    committed with four steps and no shell step at all. `usage_gate` compares
    `{{disk_check.USED_PCT}}` and no step named `disk_check` existed, so
    `condition_eval` would raise `unbound_reference` on every run and the step
    would take neither arm. `analyze_workflow` had exactly one thing to say
    about it, and it was a warning about prose.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "doomed_predicate_gate.db"
        agent_store._agent_store = None
        schedule_store._store = None
        script_store_module._script_store = None
        database.init_db()
        reset_browser_readiness_cache()

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        script_store_module._script_store = None
        database.DB_PATH = self._original_db_path
        reset_browser_readiness_cache()
        self._tmp.cleanup()

    def _commit(self, flow: list[dict], *, session=None, **extra):
        session = session or create_builder_session(system_prompt="test")
        body = {"session_id": session.session_id, "draft": _draft(flow)}
        body.update(extra)
        with mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=READY_RUNTIME),
            get_browser_runtime_readiness=mock.AsyncMock(return_value=READY_RUNTIME),
        ):
            return self.client.post("/api/agent/builder/commit", json=body)

    def _register(self, name: str, filename: str) -> dict:
        path = self.dir / filename
        path.write_text("#!/usr/bin/env bash\necho USED_PCT=1\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return script_store_module.get_script_store().register(name=name, path=str(path))

    def test_the_reported_flow_is_refused_and_names_the_unresolvable_value(self):
        response = self._commit(DISK_WATCH_FLOW)

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        self.assertEqual(payload["error"], "workflow_contract_blocked")
        self.assertTrue(payload["can_save_incomplete"])

        blocking = payload["blocking"]
        self.assertEqual(len(blocking), 1, blocking)
        finding = blocking[0]
        self.assertEqual(finding["code"], "branch_predicate_unresolvable")
        self.assertEqual(finding["step_id"], "usage_gate")
        self.assertEqual(finding["detail"]["reference"], "disk_check.USED_PCT")
        self.assertIn("{{disk_check.USED_PCT}}", finding["ask"])
        self.assertIn(finding["ask"], payload["detail"])

        # The prose heuristic is still a warning and still says what it said.
        self.assertEqual(
            [w["code"] for w in payload["warnings"]],
            ["possible_unknown_step_reference"],
        )
        self.assertEqual(agent_store.get_agent_store().count_agents(), 0)

    def test_a_satisfiable_predicate_commits(self):
        script = self._register("Check disk usage", "disk.sh")
        flow = [
            {
                "id": "disk_check",
                "type": "shell",
                "name": "Check disk usage",
                "description": "루트 사용률을 읽는다.",
                "script_id": script["id"],
            },
            *[dict(step) for step in DISK_WATCH_FLOW],
        ]
        flow[1] = {
            **DISK_WATCH_FLOW[0],
            "branches": [
                {
                    "label": "실패",
                    "when": {
                        "left": "{{steps.disk_check.exit_code}}",
                        "op": "not_equals",
                        "right": "0",
                    },
                    "target_step_id": "diagnose",
                },
                {"label": "정상", "when": None, "target_step_id": "healthy_note"},
            ],
        }
        flow[2] = {**DISK_WATCH_FLOW[1], "observation": "앞 단계의 표준 출력"}

        response = self._commit(flow)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(agent_store.get_agent_store().count_agents(), 1)
        self.assertTrue(response.json()["commit_result"]["readiness"]["ok"])

    def test_an_approved_script_no_step_runs_is_refused(self):
        session = create_builder_session(system_prompt="test")
        script = self._register("Check disk usage", "disk.sh")
        session.apply_registered_script(
            script=script, step_id="disk_check", request_name="Check disk usage"
        )
        # The approval wired a step in; this is the draft the *next* turn
        # produced, with that step gone again.
        response = self._commit(DISK_WATCH_FLOW, session=session)

        self.assertEqual(response.status_code, 400, response.text)
        codes = [f["code"] for f in response.json()["blocking"]]
        self.assertIn("approved_script_unused", codes)
        finding = next(
            f for f in response.json()["blocking"] if f["code"] == "approved_script_unused"
        )
        self.assertEqual(finding["detail"]["script_id"], script["id"])
        self.assertIn("Check disk usage", finding["ask"])
        self.assertEqual(agent_store.get_agent_store().count_agents(), 0)

    def test_the_same_session_commits_once_a_step_runs_the_script(self):
        session = create_builder_session(system_prompt="test")
        script = self._register("Check disk usage", "disk.sh")
        draft = session.apply_registered_script(
            script=script, step_id="disk_check", request_name="Check disk usage"
        )
        self.assertEqual([step.id for step in draft.flow], ["disk_check"])

        flow = [
            {
                "id": "disk_check",
                "type": "shell",
                "name": "Check disk usage",
                "description": "루트 사용률을 읽는다.",
                "script_id": script["id"],
            },
            {
                "id": "note",
                "type": "notify",
                "name": "기록",
                "description": "결과를 남긴다.",
                "on_success": {"type": "end"},
                "notify": {"title": "결과", "body": "완료", "level": "info"},
            },
        ]
        response = self._commit(flow, session=session)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(agent_store.get_agent_store().count_agents(), 1)

    def test_the_escape_hatch_still_saves_a_deliberately_unused_script(self):
        session = create_builder_session(system_prompt="test")
        script = self._register("Check disk usage", "disk.sh")
        session.apply_registered_script(
            script=script, step_id="disk_check", request_name="Check disk usage"
        )
        response = self._commit(
            [
                {
                    "id": "note",
                    "type": "notify",
                    "name": "기록",
                    "description": "결과를 남긴다.",
                    "on_success": {"type": "end"},
                    "notify": {"title": "결과", "body": "완료", "level": "info"},
                }
            ],
            session=session,
            commit_incomplete=True,
        )

        self.assertEqual(response.status_code, 200, response.text)
        readiness = response.json()["commit_result"]["readiness"]
        self.assertFalse(readiness["ok"])
        self.assertTrue(readiness["saved_incomplete"])
        self.assertIn("Check disk usage", readiness["message"])

    def test_a_session_that_approved_nothing_is_judged_exactly_as_before(self):
        response = self._commit(
            [
                {
                    "id": "note",
                    "type": "notify",
                    "name": "기록",
                    "description": "결과를 남긴다.",
                    "on_success": {"type": "end"},
                    "notify": {"title": "결과", "body": "완료", "level": "info"},
                }
            ]
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["commit_result"]["readiness"]["ok"])


if __name__ == "__main__":
    unittest.main()
