"""The workflow revision, and the write that refuses to clobber.

Two writers edit one agent's workflow — the canvas in a browser and the app's
edit screen — and before this the second Save silently won. These tests hold
the two halves of the fix to their claims:

1. **The revision names content, not time.** Identical content hashes the same
   however it was produced; a real change moves it. The reason that matters is
   in :mod:`agent.flow_revision`: the only timestamp this table has is
   second-resolution, so the timestamp's blind spot sits exactly over the race
   the precondition exists to catch.

2. **A stale precondition writes nothing.** Not "writes a bit less", not
   "writes and warns" — the store is asserted afterwards to still hold the
   other writer's version, byte for byte, and the refusal is asserted to name
   both revisions so a client can say which one it was holding.

The absent-precondition case is here too, and is not a formality: every
existing caller sends nothing, and a change that quietly made the precondition
mandatory would break the dashboard, the app's rename path and every operator
script on the day it shipped.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from agent.flow_revision import (  # noqa: E402
    FLOW_REVISION_LENGTH,
    canonical_flow_text,
    compute_flow_revision,
)
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from routes import agents as agents_routes  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class FlowRevisionHashTest(unittest.TestCase):
    """The hash itself, with no HTTP anywhere near it."""

    def test_same_content_written_differently_is_one_revision(self) -> None:
        """Key order and spacing are not content.

        SQLite hands back whatever order it stored, and a client library is
        free to serialise its JSON however it likes. If either could move the
        revision, a reader would be refused over a difference nobody made.
        """
        one = [{"id": "a", "type": "llm", "on_success": {"type": "continue"}}]
        other = [{"on_success": {"type": "continue"}, "type": "llm", "id": "a"}]
        self.assertEqual(compute_flow_revision(one), compute_flow_revision(other))

    def test_it_is_stable_across_calls(self) -> None:
        flow = [{"id": "a", "type": "llm"}]
        self.assertEqual(compute_flow_revision(flow), compute_flow_revision(flow))

    def test_a_real_change_moves_it(self) -> None:
        base = [{"id": "a", "type": "llm", "name": "Probe"}]
        renamed = [{"id": "a", "type": "llm", "name": "Probe (edited)"}]
        reordered = [{"id": "a", "type": "llm"}, {"id": "b", "type": "llm"}]
        self.assertNotEqual(compute_flow_revision(base), compute_flow_revision(renamed))
        self.assertNotEqual(
            compute_flow_revision(reordered),
            compute_flow_revision(list(reversed(reordered))),
        )

    def test_no_workflow_and_an_empty_one_are_different_versions(self) -> None:
        """Both have a revision, and they are not the same revision.

        Deleting the last step of a workflow turns one into the other. A
        reader who was told nothing had changed would be told wrong.
        """
        self.assertNotEqual(compute_flow_revision(None), compute_flow_revision([]))

    def test_non_ascii_is_hashed_as_the_text_it_is(self) -> None:
        """A Korean step name and its \\u escape are the same content."""
        self.assertIn("점검", canonical_flow_text([{"name": "점검"}]))

    def test_it_is_short_and_hex(self) -> None:
        revision = compute_flow_revision([{"id": "a"}])
        self.assertEqual(len(revision), FLOW_REVISION_LENGTH)
        int(revision, 16)  # raises if it is not hex


class _StubPairing:
    def __init__(self, paired_key: str) -> None:
        self._paired_key = paired_key

    def validate_api_key(self, api_key: str) -> bool:
        return api_key == self._paired_key


class FlowRevisionRouteTest(unittest.TestCase):
    """The revision on the read paths, and the precondition on the write."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "flow_revision.db"
        agent_store._agent_store = None
        schedule_store._store = None
        audit_store._audit_store = None

        app = FastAPI()
        app.include_router(agents_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "paired-key"
        self.client = TestClient(app)
        self.agent_id = self._create_agent()

    def tearDown(self) -> None:
        agent_store._agent_store = None
        schedule_store._store = None
        audit_store._audit_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------

    def _create_agent(self) -> str:
        response = self.client.post(
            "/api/agent/agents",
            json={
                "name": "revision-bot",
                "system_prompt": "You are useful.",
                "provider_id": "openai",
                "flow_json": [
                    {
                        "id": "probe",
                        "name": "Probe",
                        "type": "llm",
                        "on_success": {"type": "end"},
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    def _get(self) -> dict[str, Any]:
        response = self.client.get(f"/api/agent/agents/{self.agent_id}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _stored_step_names(self) -> list[str]:
        agent = agents_routes._store().get_agent(self.agent_id)
        return [step.get("name") for step in (agent or {}).get("flow_json") or []]

    def _rename_step(self, name: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "probe",
                "name": name,
                "type": "llm",
                "on_success": {"type": "end"},
            }
        ]

    # -- the read paths ----------------------------------------------------

    def test_the_single_read_publishes_it(self) -> None:
        body = self._get()
        self.assertEqual(
            body["flow_revision"],
            compute_flow_revision(body["flow_json"]),
        )

    def test_the_list_read_publishes_the_same_one(self) -> None:
        """A client that read the agent from the list can save with it.

        The list is where the app's agent screens get their agents, so a
        revision that only appeared on the single read would leave the edit
        screen with nothing to send.
        """
        listed = self.client.get("/api/agent/agents")
        self.assertEqual(listed.status_code, 200, listed.text)
        row = next(
            agent for agent in listed.json()["agents"] if agent["id"] == self.agent_id
        )
        self.assertEqual(row["flow_revision"], self._get()["flow_revision"])

    def test_renaming_a_script_does_not_move_it(self) -> None:
        """The revision is over the stored workflow, not the annotated copy.

        ``_with_script_names`` decorates the read with ``script_name`` and
        ``script_path`` looked up from the scripts table. Hashing those would
        make renaming a script refuse every pending workflow save, over a
        write to a different table that no workflow writer performed.
        """
        with_script = [{"id": "run", "type": "shell", "script_id": "s1"}]
        patched = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={"flow_json": with_script, "commit_incomplete": True},
        )
        self.assertEqual(patched.status_code, 200, patched.text)
        before = self._get()
        # The read decorated the step; the revision ignored the decoration.
        self.assertIn("script_name", before["flow_json"][0])
        self.assertEqual(
            before["flow_revision"],
            compute_flow_revision(
                agents_routes._store().get_agent(self.agent_id)["flow_json"]
            ),
        )

    # -- the write ---------------------------------------------------------

    def test_a_matching_precondition_is_applied(self) -> None:
        revision = self._get()["flow_revision"]
        response = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={
                "flow_json": self._rename_step("Probe (mine)"),
                "if_flow_revision": revision,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self._stored_step_names(), ["Probe (mine)"])
        # And the echo carries the revision the caller now holds, so a second
        # save needs no extra read.
        self.assertEqual(
            response.json()["flow_revision"],
            self._get()["flow_revision"],
        )

    def test_a_stale_precondition_writes_nothing_and_names_both_revisions(self) -> None:
        """The whole point, in one test.

        Two writers read the same revision. The first saves. The second's save
        is refused, and afterwards the store still holds *the first writer's*
        workflow — not a merge, not a partial, not the second writer's.
        """
        shared = self._get()["flow_revision"]

        first = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={
                "flow_json": self._rename_step("Probe (first writer)"),
                "if_flow_revision": shared,
            },
        )
        self.assertEqual(first.status_code, 200, first.text)
        after_first = first.json()["flow_revision"]

        second = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={
                "flow_json": self._rename_step("Probe (second writer)"),
                "if_flow_revision": shared,
            },
        )
        self.assertEqual(second.status_code, 409, second.text)
        body = second.json()
        self.assertEqual(body["error"], "flow_revision_conflict")
        self.assertEqual(body["expected_flow_revision"], shared)
        self.assertEqual(body["current_flow_revision"], after_first)
        self.assertEqual(body["agent_id"], self.agent_id)
        # A machine code *and* the full sentence, the way `unsupported_topology`
        # answers — not one or the other.
        self.assertIn("Nothing was saved", body["detail"])
        self.assertEqual(body["detail"], body["message"])

        self.assertEqual(self._stored_step_names(), ["Probe (first writer)"])

    def test_a_stale_precondition_writes_none_of_the_other_fields_either(self) -> None:
        """The refusal is about the request, not only about the workflow.

        Half-applying it would leave the agent partly one writer's and partly
        the other's, with a revision that matches neither of their reads.
        """
        shared = self._get()["flow_revision"]
        self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={
                "flow_json": self._rename_step("Probe (first writer)"),
                "if_flow_revision": shared,
            },
        )
        refused = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={
                "name": "renamed-by-the-loser",
                "flow_json": self._rename_step("Probe (second writer)"),
                "if_flow_revision": shared,
            },
        )
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(self._get()["name"], "revision-bot")

    def test_re_saving_identical_content_does_not_invent_a_conflict(self) -> None:
        """A content hash, not a timestamp, and this is what that buys.

        One writer stores byte-identical content. The revision does not move,
        so the other writer — who read before that write — is still holding
        the truth and is still allowed to save.
        """
        shared = self._get()["flow_revision"]
        identical = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={"flow_json": self._rename_step("Probe")},
        )
        self.assertEqual(identical.status_code, 200, identical.text)
        self.assertEqual(self._get()["flow_revision"], shared)

        follow_up = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={
                "flow_json": self._rename_step("Probe (mine)"),
                "if_flow_revision": shared,
            },
        )
        self.assertEqual(follow_up.status_code, 200, follow_up.text)

    def test_no_precondition_still_writes(self) -> None:
        """The compatibility affordance, held to its promise.

        Every caller that exists today sends nothing. This is the test that
        goes red if the precondition is ever quietly made mandatory.
        """
        response = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            json={"flow_json": self._rename_step("Probe (no precondition)")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self._stored_step_names(), ["Probe (no precondition)"])

    def test_a_precondition_on_a_missing_agent_is_a_404(self) -> None:
        """Not a 409: there is no version of an agent that is not there."""
        response = self.client.patch(
            "/api/agent/agents/agent_does_not_exist",
            json={"name": "x", "if_flow_revision": "0" * FLOW_REVISION_LENGTH},
        )
        self.assertEqual(response.status_code, 404, response.text)


if __name__ == "__main__":
    unittest.main()
