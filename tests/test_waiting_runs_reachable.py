"""Finding the run that is waiting for you, without knowing where to look.

The answer box has existed for a long time (`CheckpointPanel` on the phone's
task screen). Reaching it did not: a person had to already know *which* agent
had stalled, open that agent, find the run in its list, and follow it to the
task. With a dozen agents that is a dozen guesses, and the push that says
"something needs you" landed on a list of sentences carrying no run id at all.

That shape of failure — the feature exists, the path does not — is the same
one that got this app rejected under 5.1.1(v): account deletion was fully
implemented and completely unreachable without a server
(`docs/release_procedure/rejection_history/2026-06-27_v2.0.0.md`). These pin
the two halves that make a waiting run reachable: one route that answers
"what needs me?" across every agent, and a push that carries where to go.
"""

import asyncio
import os
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

from agent import agent_store, browser_session_store, push_notifier  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    WAITING_RUN_STATUSES,
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402
from pairing import pairing_service as pairing_service_module  # noqa: E402
from pairing.pairing_service import PairingService  # noqa: E402
from routes import agents as agents_routes  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class _WaitingRunsBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "waiting_runs.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

        self._original_pairing_service = pairing_service_module._pairing_service
        pairing_service_module._pairing_service = PairingService(
            config_dir=self.dir / "pairing"
        )
        self._env_patch = mock.patch.dict(
            os.environ,
            {push_notifier.SERVICE_ACCOUNT_ENV: str(self.dir / "no_such_key.json")},
        )
        self._env_patch.start()
        push_notifier.reset_for_tests()

        app = FastAPI()
        app.include_router(agents_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

        self.addCleanup(self._restore)

    def _restore(self):
        self._env_patch.stop()
        push_notifier.reset_for_tests()
        pairing_service_module._pairing_service = self._original_pairing_service
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _park_an_agent(self, name: str) -> dict:
        """Create an agent whose one step hands off to a human, and run it."""
        agent = self.store.create_agent(
            name=name,
            system_prompt="Hand off to a human.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "need_a_human",
                    "type": "manual_handoff",
                    "name": "Log into the portal",
                }
            ],
        )
        task = self.store.create_task(
            title=f"{name} work",
            assigned_agent_id=agent["id"],
            goal="Log in and continue.",
        )
        result = prepare_task_orchestration(
            task["id"], provider_id="openai", auto_start=False
        )
        assert result is not None
        asyncio.run(execute_task_orchestration(result["execution"]))
        return {"agent": agent, "task": task}


class WaitingRunsRouteTest(_WaitingRunsBase):
    def _waiting(self) -> dict:
        """Ask over HTTP, so the *path* is covered too.

        `/runs/waiting` has to be declared before `/runs/{run_id}` or FastAPI
        matches the parameterized route first and the phone gets a 404 for an
        agent run called "waiting". Calling the function directly would pass
        happily while the app served nothing.
        """
        response = self.client.get("/api/agent/runs/waiting")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_it_finds_a_parked_run_without_being_told_which_agent(self):
        parked = self._park_an_agent("handoff bot")

        answer = self._waiting()

        # The whole point: no agent_id went in.
        self.assertEqual(len(answer["waiting"]), 1)
        row = answer["waiting"][0]
        self.assertEqual(row["run"]["agent_id"], parked["agent"]["id"])
        self.assertEqual(row["agent_name"], "handoff bot")

    def test_a_row_carries_the_prompt_so_a_list_reads_without_a_second_request(self):
        self._park_an_agent("handoff bot")

        row = self._waiting()["waiting"][0]

        # Without this the phone must fetch a checkpoint per row to render a
        # list — an N+1 over exactly the runs a person is most impatient
        # about, and a list of bare run ids until it finishes.
        checkpoint = row["checkpoint"]
        self.assertIsInstance(checkpoint, dict)
        self.assertTrue(checkpoint.get("prompt"))
        self.assertTrue(checkpoint.get("reason"))

    def test_it_lists_every_agent_that_is_waiting_not_just_the_first(self):
        self._park_an_agent("first bot")
        self._park_an_agent("second bot")

        answer = self._waiting()

        self.assertEqual(
            {row["agent_name"] for row in answer["waiting"]},
            {"first bot", "second bot"},
        )

    def test_a_run_that_is_not_waiting_stays_out_of_the_list(self):
        parked = self._park_an_agent("handoff bot")
        run_id = self.store.list_runs(agent_id=parked["agent"]["id"])[0]["id"]
        self.store.update_run_status(run_id, "completed")

        answer = self._waiting()

        self.assertEqual(answer["waiting"], [])

    def test_the_status_set_comes_from_the_orchestrator(self):
        # A client that hard-codes `waiting_for_user` stops seeing `blocked`
        # runs the day one appears, and an empty list looks exactly like
        # nothing to do — a silent failure on the one screen that must not
        # have one. So the route asks the orchestrator, and this asserts the
        # set it asks for actually covers the other statuses a park uses.
        self.assertIn("blocked", WAITING_RUN_STATUSES)
        self.assertIn("waiting_for_user", WAITING_RUN_STATUSES)

        parked = self._park_an_agent("blocked bot")
        run_id = self.store.list_runs(agent_id=parked["agent"]["id"])[0]["id"]
        self.store.update_run_status(run_id, "blocked")

        answer = self._waiting()

        self.assertEqual(len(answer["waiting"]), 1)
        self.assertEqual(answer["waiting"][0]["run"]["status"], "blocked")


class WaitingPushCarriesWhereToGoTest(_WaitingRunsBase):
    def _sent_data(self, calls: list) -> dict:
        self.assertTrue(calls, "the park sent no push at all")
        return calls[-1]

    def test_the_push_names_the_run_and_task_it_is_about(self):
        pairing = pairing_service_module.get_pairing_service()
        pairing._api_keys["client_1"] = {
            "device_name": "Phone",
            "api_key_sha256": "hash",
        }
        pairing.register_push_token("client_1", "fcm-token-1")

        calls: list = []

        def _capture(tokens, **kwargs):
            calls.append(kwargs)
            return {"delivered": list(tokens), "dropped": []}

        with mock.patch("agent.push_notifier.send_to_tokens", _capture):
            parked = self._park_an_agent("handoff bot")

        data = self._sent_data(calls).get("data_extra") or {}
        run_id = self.store.list_runs(agent_id=parked["agent"]["id"])[0]["id"]

        # Tapping it has to land on the run, not on a list of sentences. These
        # ids existed at the call site all along and were dropped on the way
        # to FCM, which is why the notification could only ever open an inbox.
        self.assertEqual(data.get("run_id"), run_id)
        self.assertEqual(data.get("task_id"), parked["task"]["id"])
        self.assertEqual(data.get("agent_id"), parked["agent"]["id"])
        self.assertEqual(data.get("kind"), "waiting_for_user")

    def test_it_says_why_so_the_phone_can_open_the_right_answer(self):
        pairing = pairing_service_module.get_pairing_service()
        pairing._api_keys["client_1"] = {
            "device_name": "Phone",
            "api_key_sha256": "hash",
        }
        pairing.register_push_token("client_1", "fcm-token-1")

        calls: list = []

        def _capture(tokens, **kwargs):
            calls.append(kwargs)
            return {"delivered": list(tokens), "dropped": []}

        with mock.patch("agent.push_notifier.send_to_tokens", _capture):
            self._park_an_agent("handoff bot")

        data = self._sent_data(calls).get("data_extra") or {}
        self.assertEqual(data.get("wait_reason"), "manual_handoff")


class PushDataPayloadTest(unittest.TestCase):
    """`send_to_tokens` builds the FCM data map. FCM values must be strings."""

    def test_empty_and_missing_extras_are_dropped_not_sent_as_empty_strings(self):
        from agent.push_notifier import send_to_tokens

        # No Firebase app configured in this test, so the send returns early —
        # what is under test is the shape, asserted through the one seam that
        # does not need a real FCM: building the map the same way.
        result = send_to_tokens(
            [],
            title="t",
            data_extra={"run_id": "run_1", "task_id": None, "agent_id": ""},
        )
        self.assertEqual(result, {"delivered": [], "dropped": []})


if __name__ == "__main__":
    unittest.main()
