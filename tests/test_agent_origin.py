"""An agent says where it came from, and a prompt that cannot run cannot be typed.

Two kinds of agent share the ``agents`` table. One was authored in Code Bridge:
its stored prompt is the program, and editing it changes the next run. The
other was registered from a Claude Code agent definition on this machine: the
server rereads that ``.md`` at the start of every run, so the file is the
program and the stored ``system_prompt`` is only the stub from
``cli_agent_reference_prompt`` explaining as much.

The user-visible failure these tests exist to prevent:

Someone opens a registered agent, sees a prompt box, rewrites the instructions,
saves, and waits. Nothing changes — the file still runs. Worse, the save
overwrote the stub, so the one sentence that would have told them to edit the
file is gone, and the record now looks exactly like an ordinary authored agent
with a prompt that mysteriously has no effect. Nothing on ``GET /agents`` ever
distinguished the two, so no client could warn them.

So: the payload carries ``origin``; a patch that would write text no run reads
is refused *by the API*, naming the file, because the phone, the dashboard and
anything else all go through that one route; and the edits that genuinely do
reach a run still go through, because freezing the record would be a different
bug wearing the same coat.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, cli_agent_sources, schedule_store  # noqa: E402
from agent.agent_origin import (  # noqa: E402
    ORIGIN_AUTHORED,
    ORIGIN_CLI_AGENT_FILE,
    ORIGIN_UNKNOWN,
    AgentPromptNotEditableError,
    assert_patch_reaches_execution,
    resolve_agent_origin,
)
from core import database  # noqa: E402
from routes import agents as agents_routes  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

CLI_AGENT_FILE = textwrap.dedent(
    """\
    ---
    name: disk-watch
    description: Reports on disk pressure.
    tools: Read, Glob
    ---

    You are the disk watcher. Report free space and stop.
    """
)


class AgentOriginTestBase(unittest.TestCase):
    """One authored agent and one registered from a real file on disk."""

    def setUp(self):
        self._files = tempfile.TemporaryDirectory()
        self.root = Path(self._files.name)
        self.source = (self.root / "watch.md").resolve()
        self.source.write_text(CLI_AGENT_FILE, encoding="utf-8")

        self._db = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._db.name) / "agent_origin.db"
        agent_store._agent_store = None
        schedule_store._store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

        self._locations = mock.patch.object(
            cli_agent_sources,
            "_all_source_locations",
            return_value=[("user", self.root)],
        )
        self._locations.start()

        self.file_backed = cli_agent_sources.import_cli_agent(str(self.source)).agent
        self.authored = self.store.create_agent(
            name="nightly summary",
            description="Written here.",
            system_prompt="You are useful.",
            flow_json=[
                {"id": "step_one", "name": "Plan", "instruction": "Summarise."}
            ],
        )

        app = FastAPI()
        app.include_router(agents_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        self._locations.stop()
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        self._db.cleanup()
        self._files.cleanup()

    def _get(self, agent_id: str) -> dict:
        response = self.client.get(f"/api/agent/agents/{agent_id}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _patch(self, agent_id: str, body: dict):
        return self.client.patch(f"/api/agent/agents/{agent_id}", json=body)


class OriginOnThePayloadTest(AgentOriginTestBase):
    """Every client reads the difference off one field, or it cannot see it."""

    def test_a_registered_agent_is_marked_with_the_file_it_runs(self):
        payload = self._get(self.file_backed["id"])

        self.assertEqual(
            payload["origin"],
            {
                "kind": ORIGIN_CLI_AGENT_FILE,
                "prompt_editable": False,
                "source_path": str(self.source),
            },
        )

    def test_an_authored_agent_is_not_marked(self):
        payload = self._get(self.authored["id"])

        self.assertEqual(
            payload["origin"],
            {
                "kind": ORIGIN_AUTHORED,
                "prompt_editable": True,
                "source_path": None,
            },
        )

    def test_the_listing_carries_it_too(self):
        """The roster is where a client decides what an agent row may offer."""
        response = self.client.get("/api/agent/agents")
        self.assertEqual(response.status_code, 200, response.text)

        by_id = {agent["id"]: agent for agent in response.json()["agents"]}

        self.assertEqual(
            by_id[self.file_backed["id"]]["origin"]["kind"], ORIGIN_CLI_AGENT_FILE
        )
        self.assertEqual(by_id[self.authored["id"]]["origin"]["kind"], ORIGIN_AUTHORED)

    def test_a_failed_lookup_is_unknown_and_not_editable(self):
        """"I could not check" must never render as "yes, go ahead".

        Reporting `authored` on a failed read would hand a client an editable
        prompt box on evidence the server does not have — which is the exact
        defect, with a database hiccup as the trigger.
        """
        with mock.patch(
            "agent.agent_origin.find_import_source_path_for_agent",
            side_effect=RuntimeError("database is locked"),
        ):
            origin = resolve_agent_origin(self.file_backed["id"])

        self.assertEqual(origin.kind, ORIGIN_UNKNOWN)
        self.assertFalse(origin.prompt_editable)
        self.assertIsNone(origin.source_path)

    def test_an_unrecognised_future_origin_still_answers_the_only_question(self):
        """`prompt_editable` is what an un-updated client reads.

        A third origin later is a new `kind` value. A shipped app that has
        never heard of it must still not offer a prompt box, so the boolean is
        carried beside the kind rather than derived from it.
        """
        payload = self._get(self.file_backed["id"])

        self.assertIn("prompt_editable", payload["origin"])
        self.assertIs(payload["origin"]["prompt_editable"], False)


class RefusingEditsThatWouldNotRunTest(AgentOriginTestBase):
    """The guard is on the API, because that is what every client shares."""

    def test_patching_the_prompt_of_a_file_backed_agent_is_refused(self):
        response = self._patch(
            self.file_backed["id"],
            {"system_prompt": "You are the NEW disk watcher. Do something else."},
        )

        self.assertEqual(response.status_code, 409, response.text)
        body = response.json()
        self.assertEqual(body["error"], "agent_prompt_not_editable")
        self.assertIn(str(self.source), body["detail"])
        self.assertEqual(body["origin"]["source_path"], str(self.source))

    def test_a_refused_edit_leaves_the_stub_intact(self):
        """The stub is the last line of defence for any client not yet updated.

        Destroying it was the second half of the original defect: after the
        overwrite, edits still did nothing *and* nothing said why.
        """
        before = self.store.get_agent(self.file_backed["id"])["system_prompt"]

        self._patch(self.file_backed["id"], {"system_prompt": "anything at all"})

        after = self.store.get_agent(self.file_backed["id"])["system_prompt"]
        self.assertEqual(after, before)
        self.assertIn("Editing this text changes nothing", after)

    def test_patching_a_step_instruction_is_refused_naming_the_file(self):
        """An llm step's instruction is never relayed for a file-backed agent.

        `task_orchestrator._workflow_step_message` prints a pointer at the file
        instead, so text typed here is as inert as the system prompt.
        """
        flow = self.store.get_agent(self.file_backed["id"])["flow_json"]
        edited = [
            {**step, "instruction": "Ignore the file and do this instead."}
            for step in flow
        ]

        response = self._patch(self.file_backed["id"], {"flow_json": edited})

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn(str(self.source), response.json()["detail"])

    def test_a_new_step_carrying_an_instruction_is_refused_too(self):
        """Adding a step does not create a second, working prompt box."""
        flow = self.store.get_agent(self.file_backed["id"])["flow_json"]

        response = self._patch(
            self.file_backed["id"],
            {
                "flow_json": [
                    *flow,
                    {
                        "id": "step_extra",
                        "name": "Follow up",
                        "instruction": "Then email the result.",
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 409, response.text)

    def test_patching_the_prompt_of_an_authored_agent_still_works(self):
        response = self._patch(
            self.authored["id"], {"system_prompt": "You are extremely useful."}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.store.get_agent(self.authored["id"])["system_prompt"],
            "You are extremely useful.",
        )

    def test_patching_an_authored_step_instruction_still_works(self):
        response = self._patch(
            self.authored["id"],
            {
                "flow_json": [
                    {
                        "id": "step_one",
                        "name": "Plan",
                        "instruction": "Summarise yesterday instead.",
                    }
                ]
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["flow_json"][0]["instruction"],
            "Summarise yesterday instead.",
        )


class WhatStaysEditableTest(AgentOriginTestBase):
    """Not a freeze. The workflow around a file-backed agent is still the user's."""

    def test_the_name_and_description_can_still_be_changed(self):
        """The name that selects the agent in the session comes from the file.

        `cli_agent_runtime._to_definition` reads it there, so the record's name
        is a label — renaming it cannot desynchronise a run.
        """
        response = self._patch(
            self.file_backed["id"],
            {"name": "disk watcher (nightly)", "description": "Runs at 3am."},
        )

        self.assertEqual(response.status_code, 200, response.text)
        stored = self.store.get_agent(self.file_backed["id"])
        self.assertEqual(stored["name"], "disk watcher (nightly)")
        self.assertEqual(stored["description"], "Runs at 3am.")

    def test_the_rest_of_a_step_can_still_be_changed(self):
        """title / success_criteria / on_failure *are* relayed to the run.

        Refusing them would freeze the only part of a file-backed agent Code
        Bridge actually contributes.
        """
        flow = self.store.get_agent(self.file_backed["id"])["flow_json"]
        edited = [
            {
                **step,
                "name": "Check the disk",
                "success_criteria": "Free space reported for every volume.",
            }
            for step in flow
        ]

        response = self._patch(self.file_backed["id"], {"flow_json": edited})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["flow_json"][0]["name"], "Check the disk")

    def test_a_shell_or_notify_step_can_still_be_added(self):
        """A notify step runs in full — it is not the definition's prompt."""
        flow = self.store.get_agent(self.file_backed["id"])["flow_json"]

        response = self._patch(
            self.file_backed["id"],
            {
                "flow_json": [
                    *flow,
                    {
                        "id": "step_tell_me",
                        "type": "notify",
                        "name": "Tell me",
                        "notify": {"title": "Disk report", "level": "info"},
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["flow_json"]), len(flow) + 1)

    def test_resending_the_unchanged_prompt_is_not_an_edit(self):
        """The app and the dashboard both send the whole form on save.

        Refusing the mere presence of `system_prompt` would make renaming a
        file-backed agent impossible from either — a freeze, not a guard.
        """
        stored = self.store.get_agent(self.file_backed["id"])

        response = self._patch(
            self.file_backed["id"],
            {"name": "renamed", "system_prompt": stored["system_prompt"]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.store.get_agent(self.file_backed["id"])["name"], "renamed")


class DashboardSharesTheGuardTest(AgentOriginTestBase):
    """The dashboard edits agents too, through the same handlers.

    It delegates to ``routes.agents`` rather than reimplementing, which is what
    makes "the guard is on the API" true rather than aspirational — but only
    while it keeps delegating, so this pins it.
    """

    def setUp(self):
        super().setUp()
        from routes import dashboard_agents
        from routes.deps import require_local_access

        app = FastAPI()
        app.include_router(dashboard_agents.router)
        app.dependency_overrides[require_local_access] = lambda: None
        self.dashboard = TestClient(app)

    def test_the_dashboard_sees_the_origin(self):
        response = self.dashboard.get(
            f"/api/dashboard/agent/agents/{self.file_backed['id']}"
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["origin"]["source_path"], str(self.source)
        )

    def test_the_dashboard_is_refused_the_same_edit(self):
        response = self.dashboard.patch(
            f"/api/dashboard/agent/agents/{self.file_backed['id']}",
            json={"system_prompt": "rewritten from the PC"},
        )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn(str(self.source), response.json()["detail"])


class DashboardTemplateShowsTheDifferenceTest(unittest.TestCase):
    """The dashboard's own editor must lock the prompt, not just the API.

    The API refusal is what makes the guard real, but a dashboard that still
    presents a live prompt box teaches the user to type into it and then hands
    them a 409 they did not ask for. This reads the shipped template because
    there is no other way to hold a page of inline JavaScript to a behaviour —
    and the behaviour is small and named: `applyAgentOrigin` is the one place
    the editor decides.
    """

    @classmethod
    def setUpClass(cls):
        cls.template = (
            SERVER_DIR / "dashboard" / "templates" / "agents.html"
        ).read_text(encoding="utf-8")

    def test_the_editor_locks_the_prompt_from_the_origin(self):
        self.assertIn("function applyAgentOrigin(origin)", self.template)
        self.assertIn("prompt.readOnly = !editable", self.template)
        # Called with what the server actually sent for this agent.
        self.assertIn("applyAgentOrigin(agent.origin)", self.template)

    def test_an_absent_origin_locks_rather_than_unlocks(self):
        """Same fail-closed rule as the app: no answer is not permission."""
        self.assertIn("agentOrigin.prompt_editable === true", self.template)
        self.assertIn(
            "{ kind: 'unknown', prompt_editable: false, source_path: null }",
            self.template,
        )

    def test_a_locked_step_instruction_is_readonly_too(self):
        self.assertIn("function stepFieldLocked(step, fieldKey)", self.template)
        self.assertIn("const lockAttr = locked ? ' readonly' : ''", self.template)

    def test_a_locked_prompt_is_not_sent_back(self):
        self.assertIn("if (originPromptEditable()) {", self.template)

    def test_the_strings_exist_in_both_dashboard_locales(self):
        for key in (
            "origin_file",
            "origin_prompt_locked",
            "origin_unknown",
            "origin_unknown_locked",
        ):
            # Once under `en`, once under `ko`.
            self.assertEqual(
                self.template.count(f"{key}: '"), 2, f"{key} is not in both locales"
            )


class GuardUnitTest(unittest.TestCase):
    """The guard's own decisions, without a database or a route in the way."""

    def _refuses(self, patch: dict, agent: dict | None = None) -> str:
        from agent.agent_origin import AgentOrigin

        origin = AgentOrigin(
            kind=ORIGIN_CLI_AGENT_FILE,
            prompt_editable=False,
            source_path="/Users/someone/.claude/agents/watch.md",
        )
        with self.assertRaises(AgentPromptNotEditableError) as caught:
            assert_patch_reaches_execution(
                agent=agent or {"system_prompt": "stub", "flow_json": []},
                patch=patch,
                origin=origin,
            )
        self.assertIs(caught.exception.origin, origin)
        return str(caught.exception)

    def test_an_unknown_origin_is_guarded_like_a_file_backed_one(self):
        """Failing closed is the whole point of having an `unknown` kind."""
        from agent.agent_origin import UNKNOWN_ORIGIN

        with self.assertRaises(AgentPromptNotEditableError) as caught:
            assert_patch_reaches_execution(
                agent={"system_prompt": "stub"},
                patch={"system_prompt": "something new"},
                origin=UNKNOWN_ORIGIN,
            )

        # No file to name, so it says what it does know rather than inventing one.
        self.assertIn("agent definition file", str(caught.exception))

    def test_an_authored_origin_is_not_guarded_at_all(self):
        from agent.agent_origin import AUTHORED_ORIGIN

        assert_patch_reaches_execution(
            agent={"system_prompt": "old", "flow_json": []},
            patch={"system_prompt": "new", "flow_json": [{"id": "a", "instruction": "x"}]},
            origin=AUTHORED_ORIGIN,
        )

    def test_the_message_names_the_file(self):
        message = self._refuses({"system_prompt": "new text"})

        self.assertIn("/Users/someone/.claude/agents/watch.md", message)

    def test_a_shell_steps_fields_are_not_mistaken_for_a_prompt(self):
        """A shell step runs a registered script; `instruction` is not relayed."""
        from agent.agent_origin import AgentOrigin

        assert_patch_reaches_execution(
            agent={"system_prompt": "stub", "flow_json": []},
            patch={
                "flow_json": [
                    {
                        "id": "s1",
                        "type": "shell",
                        "script_id": "sc_1",
                        "instruction": "leftover text",
                    }
                ]
            },
            origin=AgentOrigin(
                kind=ORIGIN_CLI_AGENT_FILE,
                prompt_editable=False,
                source_path="/tmp/watch.md",
            ),
        )

    def test_removing_an_instruction_entirely_is_allowed(self):
        """Deleting inert text is not writing inert text."""
        from agent.agent_origin import AgentOrigin

        assert_patch_reaches_execution(
            agent={"system_prompt": "stub", "flow_json": [{"id": "a", "instruction": "x"}]},
            patch={"flow_json": [{"id": "a", "name": "Step"}]},
            origin=AgentOrigin(
                kind=ORIGIN_CLI_AGENT_FILE,
                prompt_editable=False,
                source_path="/tmp/watch.md",
            ),
        )


if __name__ == "__main__":
    unittest.main()
