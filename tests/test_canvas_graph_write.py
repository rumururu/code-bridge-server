"""The canvas save path, and the three refusals it has to be able to show.

T-I1-14 adds no server route. It writes through
``AgentUpdateBody.flow_graph``, which every agent door already accepts, and it
draws whatever comes back. So the thing worth testing here is not the plumbing
— :mod:`tests.test_canvas_session_token` already round-trips one graph — but
the two claims the canvas is built on top of:

1. **The graphs the browser produces are graphs this server folds.** The canvas
   derives its edges in TypeScript, by mirroring ``_derive_edges``
   (``frontend/packages/flow-canvas/src/codeBridge/linearEdits.ts``). A mirror
   is free to drift, and the cost of drift is total: ``from_graph`` re-derives
   the edges from the policies and compares, so a client whose derivation
   disagreed would be refused on every save it ever attempted. The graphs below
   are what that TypeScript emits for the four edits the canvas offers — add,
   delete, rename, set a policy — written out here and fed to the real fold.

2. **Every refusal arrives whole.** ``unsupported_topology`` carries the *full*
   issue list on purpose (``flow_graph.py``: all violations at once, never
   fail-fast on the first). The canvas renders that list rather than a count,
   which is only worth doing if the list actually arrives — so the shape the
   browser reads is asserted here, field by field, against the response the
   route really produces.

The kernel-absent case is simulated rather than skipped: it is the one refusal
that cannot be produced on a machine that *has* the kernel, and it is also the
one a real deployment hits most often (the deployed venv has no
``agent_flow_core`` — see ``_fold_flow_graph_input``).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from agent.flow_revision import compute_flow_revision  # noqa: E402
from audit import audit_store  # noqa: E402
from canvas.canvas_access import get_canvas_session_manager  # noqa: E402
from core import database  # noqa: E402
from pairing import pairing_service as pairing_service_module  # noqa: E402
from routes import agents as agents_routes  # noqa: E402
from routes import canvas_api  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

KERNEL_PRESENT = importlib.util.find_spec("agent_flow_core") is not None


def _step(
    step_id: str,
    name: str,
    *,
    on_success: dict[str, Any] | None = None,
    on_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One step in kernel wire shape, as the canvas sends it.

    ``onSuccess``/``onFailure`` are omitted when ``None`` rather than sent as
    null, because that is what the canvas does for a step it just added: an
    unset slot is the server's default, and the browser does not restate a
    default it does not own.
    """
    step: dict[str, Any] = {
        "id": step_id,
        "stepType": "llm",
        "name": name,
        "description": "",
        "config": {},
    }
    if on_success is not None:
        step["onSuccess"] = on_success
    if on_failure is not None:
        step["onFailure"] = on_failure
    return step


def _edge(from_id: str, to_id: str, on: str, kind: str, via: str | None = None) -> dict:
    """One edge exactly as ``deriveLinearEdges`` emits it."""
    annotation: dict[str, Any] = {"on": on}
    if via is not None:
        annotation["via"] = via
    return {
        "id": f"{from_id}:{on}",
        "fromStepId": from_id,
        "toStepId": to_id,
        "fromField": None,
        "toField": None,
        "kind": kind,
        "extensions": {"codeBridgeLinear": annotation},
    }


class _StubPairing:
    def __init__(self, paired_key: str) -> None:
        self._paired_key = paired_key

    def validate_api_key(self, api_key: str) -> bool:
        return api_key == self._paired_key


class CanvasGraphWriteTest(unittest.TestCase):
    PAIRED_KEY = "paired-key-for-graph-write-tests"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "canvas_graph_write.db"
        agent_store._agent_store = None
        schedule_store._store = None
        audit_store._audit_store = None
        get_canvas_session_manager().clear()

        self._original_pairing = pairing_service_module._pairing_service
        pairing_service_module._pairing_service = _StubPairing(self.PAIRED_KEY)

        app = FastAPI()
        app.include_router(agents_routes.router)
        app.include_router(canvas_api.session_router)
        app.include_router(canvas_api.router)
        app.dependency_overrides[verify_api_key] = lambda: self.PAIRED_KEY
        self.app = app
        self.client = TestClient(app)

        self.agent_id = self._create_agent()
        self.token = self._issue()

    def tearDown(self) -> None:
        pairing_service_module._pairing_service = self._original_pairing
        get_canvas_session_manager().clear()
        agent_store._agent_store = None
        schedule_store._store = None
        audit_store._audit_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------

    def _create_agent(self) -> str:
        """Three steps, and two of them jump to the third when they fail.

        The shape the ticket is about: deleting ``review`` is one tap, and it
        leaves two policies naming a step that is gone.
        """
        response = self.client.post(
            "/api/agent/agents",
            json={
                "name": "canvas-write-bot",
                "system_prompt": "You are useful.",
                "provider_id": "openai",
                "flow_json": [
                    {
                        "id": "probe",
                        "name": "Probe",
                        "type": "llm",
                        "on_success": {"type": "continue"},
                        "on_failure": {
                            "type": "goto_step",
                            "target_step_id": "review",
                        },
                    },
                    {
                        "id": "apply",
                        "name": "Apply",
                        "type": "llm",
                        "on_success": {"type": "end"},
                        "on_failure": {
                            "type": "goto_step",
                            "target_step_id": "review",
                        },
                    },
                    {
                        "id": "review",
                        "name": "Review",
                        "type": "llm",
                        "on_success": {"type": "end"},
                        "on_failure": {"type": "ask_user", "resume": "same_step"},
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    def _issue(self) -> str:
        response = self.client.post(
            "/api/agent/canvas/session", json={"agent_id": self.agent_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["token"]

    def _auth(self) -> dict[str, str]:
        return {canvas_api.CANVAS_TOKEN_HEADER: self.token}

    def _read_graph(self) -> dict[str, Any]:
        response = self.client.get(
            f"/api/canvas/agents/{self.agent_id}/graph", headers=self._auth()
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("flow_graph", body, body)
        return body["flow_graph"]

    def _save(self, flow_graph: Any, *, if_flow_revision: str | None = None):
        body: dict[str, Any] = {"flow_graph": flow_graph}
        # Omitted rather than sent as null when there is none, so the "no
        # precondition" tests exercise the shape an older bundle really sends.
        if if_flow_revision is not None:
            body["if_flow_revision"] = if_flow_revision
        return self.client.patch(
            f"/api/canvas/agents/{self.agent_id}/graph",
            headers=self._auth(),
            json=body,
        )

    def _graph_revision(self) -> str:
        response = self.client.get(
            f"/api/canvas/agents/{self.agent_id}/graph", headers=self._auth()
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["flow_revision"]

    def _stored_step_ids(self) -> list[str]:
        agent = agents_routes._store().get_agent(self.agent_id)
        return [step["id"] for step in (agent or {}).get("flow_json") or []]

    # -- what the browser sends is what this server folds -------------------

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_the_read_graph_is_savable_unchanged(self) -> None:
        """The floor: reading and saving without editing changes nothing.

        Not trivial. It is the assertion that the wire form the canvas is
        handed is one the write gate accepts, so any later refusal is about
        the *edit* and not about the round trip itself.
        """
        graph = self._read_graph()
        response = self._save(graph)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self._stored_step_ids(), ["probe", "apply", "review"])

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_a_step_added_by_the_canvas_folds_and_is_stored(self) -> None:
        """`addStep` inserts after the selection and derives the two edges.

        The added step carries **no policies at all** — the canvas does not
        restate the kernel's defaults — and the sequential edges either side of
        it are derived from the *neighbours'* policies. If the server disagreed
        about either, this is where it would say so.
        """
        graph = self._read_graph()
        graph["steps"].insert(1, _step("llm", "Added"))
        graph["edges"] = [
            _edge("probe", "llm", "success", "seq"),
            _edge("probe", "review", "failure", "goto"),
            _edge("llm", "apply", "success", "seq"),
            _edge("apply", "review", "failure", "goto"),
        ]
        response = self._save(graph)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self._stored_step_ids(), ["probe", "llm", "apply", "review"]
        )
        # And the graph the canvas now draws is the server's, not its own.
        self.assertEqual(
            [step["id"] for step in response.json()["flow_graph"]["steps"]],
            ["probe", "llm", "apply", "review"],
        )

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_a_renamed_step_is_stored_under_its_new_name(self) -> None:
        graph = self._read_graph()
        graph["steps"][0]["name"] = "Probe (renamed)"
        response = self._save(graph)
        self.assertEqual(response.status_code, 200, response.text)
        agent = agents_routes._store().get_agent(self.agent_id)
        self.assertEqual(agent["flow_json"][0]["name"], "Probe (renamed)")

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_replacing_a_jump_with_a_value_drops_the_edge_with_it(self) -> None:
        """The picker's other half: a policy that draws nothing.

        The canvas removes the edge because it re-derives *all* of them from
        the policies. A client that edited edges instead would leave
        ``probe:failure`` behind and be told ``linear.edge_unbacked``.
        """
        graph = self._read_graph()
        graph["steps"][0]["onFailure"] = {"type": "abort"}
        graph["edges"] = [
            _edge("probe", "apply", "success", "seq"),
            _edge("apply", "review", "failure", "goto"),
        ]
        response = self._save(graph)
        self.assertEqual(response.status_code, 200, response.text)
        agent = agents_routes._store().get_agent(self.agent_id)
        self.assertEqual(agent["flow_json"][0]["on_failure"], {"type": "abort"})

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_a_retry_chain_edge_derived_by_the_canvas_is_accepted(self) -> None:
        """E4, the derivation rule easiest to get wrong in a mirror.

        A failure ``retry`` whose ``then`` ends in a jump produces one goto
        edge annotated ``via: retry_then``. The canvas walks the nested chain
        to find it (`terminalRetryGoto`); this asserts the server agrees about
        both the edge and the annotation.
        """
        graph = self._read_graph()
        graph["steps"][0]["onFailure"] = {
            "type": "retry",
            "max_attempts": 2,
            "then": {"type": "goto_step", "target_step_id": "review"},
        }
        graph["edges"] = [
            _edge("probe", "apply", "success", "seq"),
            _edge("probe", "review", "failure", "goto", via="retry_then"),
            _edge("apply", "review", "failure", "goto"),
        ]
        response = self._save(graph)
        self.assertEqual(response.status_code, 200, response.text)

    # -- refusal 1: unsupported_topology, with every issue ------------------

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_deleting_a_jump_target_is_refused_with_every_issue_listed(self) -> None:
        """Acceptance ②: the save attempt shows *all* the server's issues.

        Two steps jump to ``review``. Removing it leaves both policies naming
        a step that is gone, and the canvas keeps deriving both edges (because
        ``_derive_edges`` does too), so the refusal is one issue per policy —
        not one issue for the first policy found.
        """
        graph = self._read_graph()
        graph["steps"] = [
            step for step in graph["steps"] if step["id"] != "review"
        ]
        graph["edges"] = [
            _edge("probe", "apply", "success", "seq"),
            _edge("probe", "review", "failure", "goto"),
            _edge("apply", "review", "failure", "goto"),
        ]
        response = self._save(graph)

        self.assertEqual(response.status_code, 400, response.text)
        body = response.json()
        self.assertEqual(body["error"], "unsupported_topology")
        self.assertTrue(body["message"])

        issues = body["issues"]
        self.assertEqual(
            [(issue["code"], issue["stepId"]) for issue in issues],
            [
                ("policy.goto_target_missing", "probe"),
                ("policy.goto_target_missing", "apply"),
            ],
        )
        # Each issue carries what the canvas prints: a sentence and the target
        # that is missing, so the panel can name it without parsing prose.
        for issue in issues:
            self.assertIn("review", issue["message"])
            self.assertEqual(issue["detail"], {"targetStepId": "review"})

        # Nothing was stored. The user's edits live in the browser; the agent
        # is exactly as it was.
        self.assertEqual(self._stored_step_ids(), ["probe", "apply", "review"])

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_an_edge_the_policies_do_not_back_is_named_as_such(self) -> None:
        """The refusal a client that edited edges directly would live in.

        Worth pinning because it is the failure mode the canvas's design
        avoids by construction: it never edits an edge, it edits a policy and
        re-derives. If that ever changed, this is the message the user would
        start seeing.
        """
        graph = self._read_graph()
        graph["edges"].append(_edge("review", "probe", "success", "goto"))
        response = self._save(graph)
        self.assertEqual(response.status_code, 400, response.text)
        codes = {issue["code"] for issue in response.json()["issues"]}
        self.assertIn("linear.edge_unbacked", codes)

    # -- refusal 2: invalid_flow_graph --------------------------------------

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_something_that_is_not_a_flow_is_refused_by_name(self) -> None:
        response = self._save({"steps": "not-a-list"})
        self.assertEqual(response.status_code, 400, response.text)
        body = response.json()
        self.assertEqual(body["error"], "invalid_flow_graph")
        # No issue list: there is no topology to have issues *with*. The canvas
        # renders the message alone, which is why it must be a real sentence.
        self.assertNotIn("issues", body)
        self.assertIn("does not validate as a kernel Flow", body["message"])
        self.assertEqual(self._stored_step_ids(), ["probe", "apply", "review"])

    # -- refusal 3: kernel_not_installed ------------------------------------

    def test_a_server_without_the_kernel_refuses_in_a_human_sentence(self) -> None:
        """Acceptance ③, on the machine where it actually happens.

        ``_fold_flow_graph_input`` imports the kernel lazily so that a server
        without it still starts and still accepts ``flow_json``. Putting
        ``None`` in ``sys.modules`` is how that same ``ImportError`` is
        reproduced on a machine that has the kernel installed — skipping the
        case here would leave the most common deployment untested.
        """
        with mock.patch.dict(sys.modules, {"agent.flow_graph": None}):
            response = self._save({"steps": [], "edges": []})

        self.assertEqual(response.status_code, 422, response.text)
        body = response.json()
        self.assertEqual(body["error"], "kernel_not_installed")
        self.assertEqual(body["reason"], "kernel_not_installed")
        # The canvas prints `message` verbatim, so it has to stand on its own:
        # what happened, that nothing was lost, and what to do instead.
        self.assertIn("no agent-flow-core kernel", body["message"])
        self.assertIn("Nothing was saved", body["message"])
        self.assertTrue(body["remedy"])
        self.assertEqual(self._stored_step_ids(), ["probe", "apply", "review"])

    # -- the refusal reaches the canvas unchanged ---------------------------

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_the_canvas_door_returns_the_refusal_verbatim(self) -> None:
        """The canvas route adds no validation and subtracts no detail.

        Both doors are asked the same bad question and must give the same
        answer, byte for byte — otherwise the canvas would be reading a
        summary of a refusal while the dashboard read the refusal.
        """
        graph = self._read_graph()
        graph["steps"] = [
            step for step in graph["steps"] if step["id"] != "review"
        ]

        through_canvas = self._save(graph)
        through_agent = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            headers={"X-API-Key": self.PAIRED_KEY},
            json={"flow_graph": graph},
        )
        self.assertEqual(through_canvas.status_code, through_agent.status_code)
        self.assertEqual(through_canvas.json(), through_agent.json())

    # -- the canvas can hold a revision, and refuse to clobber with it ------

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_the_graph_read_carries_the_revision_the_save_needs(self) -> None:
        """The graph view is narrowed by ``_GRAPH_VIEW_KEYS``, so this is real.

        A key not on that tuple never reaches the browser. The canvas is the
        writer most likely to be racing the app's edit screen, so a graph read
        that dropped the revision would leave the one client that most needs
        a precondition with nothing to send.
        """
        response = self.client.get(
            f"/api/canvas/agents/{self.agent_id}/graph", headers=self._auth()
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("flow_revision", body)
        stored = agents_routes._store().get_agent(self.agent_id)
        self.assertEqual(
            body["flow_revision"],
            compute_flow_revision((stored or {}).get("flow_json")),
        )

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_two_canvases_on_one_agent_and_the_second_save_is_refused(self) -> None:
        """The race this whole change is about, played out on the real door.

        Both read the same graph. Both edit. The first saves; the second is
        refused, and the *first writer's* step names are still what is stored
        afterwards.
        """
        graph = self._read_graph()
        shared = self._graph_revision()

        first = dict(graph)
        first["steps"] = [
            {**step, "name": "First writer"} if step["id"] == "probe" else step
            for step in graph["steps"]
        ]
        accepted = self._save(first, if_flow_revision=shared)
        self.assertEqual(accepted.status_code, 200, accepted.text)
        # The write answers with the graph view, so the revision the canvas
        # now holds arrives with it rather than needing another read.
        self.assertNotEqual(accepted.json()["flow_revision"], shared)

        second = dict(graph)
        second["steps"] = [
            {**step, "name": "Second writer"} if step["id"] == "probe" else step
            for step in graph["steps"]
        ]
        refused = self._save(second, if_flow_revision=shared)
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(refused.json()["error"], "flow_revision_conflict")
        self.assertEqual(refused.json()["expected_flow_revision"], shared)

        stored = agents_routes._store().get_agent(self.agent_id) or {}
        self.assertEqual(
            [step.get("name") for step in stored.get("flow_json") or []],
            ["First writer", "Apply", "Review"],
        )

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_a_canvas_save_without_a_precondition_still_works(self) -> None:
        """A bundle older than this change keeps saving.

        The compatibility affordance reaches the canvas door too, and it has
        to: the bundle is a build artifact committed into a different
        repository, so an installed server can legitimately be serving one
        that predates the field.
        """
        graph = self._read_graph()
        self.assertEqual(self._save(graph).status_code, 200)

    @unittest.skipUnless(KERNEL_PRESENT, "agent-flow-core kernel not installed")
    def test_the_conflict_reaches_the_canvas_verbatim_too(self) -> None:
        """Same discipline as the topology refusal: no summarising at the door."""
        graph = self._read_graph()
        stale = "0" * 16
        through_canvas = self._save(graph, if_flow_revision=stale)
        through_agent = self.client.patch(
            f"/api/agent/agents/{self.agent_id}",
            headers={"X-API-Key": self.PAIRED_KEY},
            json={"flow_graph": graph, "if_flow_revision": stale},
        )
        self.assertEqual(through_canvas.status_code, 409)
        self.assertEqual(through_canvas.status_code, through_agent.status_code)
        self.assertEqual(through_canvas.json(), through_agent.json())


if __name__ == "__main__":
    unittest.main()
