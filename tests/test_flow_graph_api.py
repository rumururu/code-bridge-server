"""The read API's derived graph view (agent-flow-core T-B-04).

``GET /agents/{id}`` carries ``flow_graph``: the kernel view of the stored
linear workflow, derived through ``agent.flow_graph.to_graph``. Two things
are pinned here beyond "the field is present".

1. **The server survives a missing kernel.** The deployed venv
   (``~/.code-bridge/venv``) has no ``agent_flow_core`` installed, and
   ``agent/flow_graph.py`` imports it at module top. The route therefore
   imports the converter *inside* the handler and answers 200 with
   ``flow_graph_unavailable`` when the import fails.
   ``KernelMissingTest`` simulates exactly that condition — it makes
   ``import agent_flow_core`` raise the way an uninstalled package does —
   so this stays true without anyone having to uninstall anything.

2. **A missing graph always says why.** Silence (an absent field, or
   ``flow_graph: null``) reads as "this agent has no graph", which is a claim
   about the agent when the truth is a claim about this server or about a
   stored workflow that does not fold. Every no-graph path here asserts a
   ``reason``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from core import database  # noqa: E402
from routes import agents, dashboard_agents  # noqa: E402
from routes.deps import require_local_access, verify_api_key  # noqa: E402

LINEAR_FLOW = [
    {"id": "plan", "type": "llm", "name": "Plan", "instruction": "decide"},
    {
        "id": "tell",
        "type": "notify",
        "name": "Tell",
        "notify": {"title": "done", "body": "finished"},
    },
]


@contextmanager
def kernel_uninstalled():
    """Run the block as if ``agent_flow_core`` were not installed.

    A ``None`` entry in ``sys.modules`` is what the import machinery treats
    as "this import is halted" and raises ``ImportError`` for — the same
    exception class an absent distribution produces. ``agent.flow_graph`` is
    evicted alongside it so the route's lazy import has to re-execute the
    module (and hit its top-level kernel import) rather than find the copy
    this test process already loaded.
    """

    blocked = [
        name
        for name in list(sys.modules)
        if name == "agent_flow_core"
        or name.startswith("agent_flow_core.")
        or name == "agent.flow_graph"
    ]
    saved = {name: sys.modules[name] for name in blocked}
    for name in blocked:
        del sys.modules[name]
    sys.modules["agent_flow_core"] = None
    try:
        yield
    finally:
        sys.modules.pop("agent_flow_core", None)
        sys.modules.pop("agent.flow_graph", None)
        sys.modules.update(saved)


class FlowGraphApiTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_flow_graph_api.db"
        agent_store._agent_store = None
        schedule_store._store = None
        database.init_db()

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

        dashboard_app = FastAPI()
        dashboard_app.include_router(dashboard_agents.router)
        dashboard_app.dependency_overrides[require_local_access] = lambda: None
        self.dashboard = TestClient(dashboard_app)

        self.store = agent_store.get_agent_store()

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _store_agent(self, flow_json, name: str = "grapher") -> str:
        """Write an agent straight to the store.

        Deliberately not through ``POST /agents``: the write path normalizes
        (and refuses) workflows, and one of the cases below is precisely a
        stored workflow that no longer normalizes — a row a live database can
        hold and this route still has to answer for.
        """

        agent = self.store.create_agent(
            name=name,
            description="flow graph fixture",
            system_prompt="You are useful.",
            provider_id="openai",
            flow_json=flow_json,
        )
        return str(agent["id"])


class FlowGraphPresentTest(FlowGraphApiTestBase):
    """(a) With the kernel installed, the single read carries the graph."""

    def test_single_read_carries_steps_and_edges(self):
        agent_id = self._store_agent(LINEAR_FLOW)

        response = self.client.get(f"/api/agent/agents/{agent_id}")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertNotIn("flow_graph_unavailable", payload)
        graph = payload["flow_graph"]
        self.assertEqual(
            [(step["id"], step["stepType"]) for step in graph["steps"]],
            [("plan", "llm"), ("tell", "notify")],
        )
        # One derived edge: `plan`'s default success policy is `continue`, and
        # the trailing step's produces nothing (spec section 3/5).
        self.assertEqual(
            [
                (edge["id"], edge["fromStepId"], edge["toStepId"], edge["kind"])
                for edge in graph["edges"]
            ],
            [("plan:success", "plan", "tell", "seq")],
        )
        self.assertEqual(
            graph["edges"][0]["extensions"]["codeBridgeLinear"], {"on": "success"}
        )

    def test_the_graph_is_derived_from_the_canon_not_the_annotated_copy(self):
        """``flow_json`` keeps being the canon, and stays whole beside it."""

        agent_id = self._store_agent(LINEAR_FLOW)

        payload = self.client.get(f"/api/agent/agents/{agent_id}").json()

        self.assertEqual(
            [step["id"] for step in payload["flow_json"]], ["plan", "tell"]
        )
        self.assertEqual(payload["id"], agent_id)
        self.assertEqual(payload["name"], "grapher")

    def test_the_list_route_stays_off_the_derivation(self):
        """The graph is a detail-view field, by decision — see
        ``_agent_with_next_fire``: the list serves up to 200 agents and draws
        names, not shapes."""

        self._store_agent(LINEAR_FLOW)

        listed = self.client.get("/api/agent/agents").json()["agents"]

        self.assertEqual(len(listed), 1)
        self.assertNotIn("flow_graph", listed[0])
        self.assertNotIn("flow_graph_unavailable", listed[0])


class KernelMissingTest(FlowGraphApiTestBase):
    """(b) No kernel: the read still answers, and says what is missing."""

    def test_the_import_really_fails_under_the_simulation(self):
        """The simulation is only worth anything if it reproduces the fault."""

        with kernel_uninstalled():
            with self.assertRaises(ImportError):
                from agent.flow_graph import to_graph  # noqa: F401

    def test_read_stays_200_and_names_the_reason(self):
        agent_id = self._store_agent(LINEAR_FLOW)

        with kernel_uninstalled():
            response = self.client.get(f"/api/agent/agents/{agent_id}")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertNotIn("flow_graph", payload)
        unavailable = payload["flow_graph_unavailable"]
        self.assertEqual(unavailable["reason"], "kernel_not_installed")
        self.assertIn("agent-flow-core", unavailable["message"])
        # Not a bare flag: the message has to leave the reader knowing the
        # workflow itself is fine.
        self.assertIn("flow_json", unavailable["message"])

    def test_the_message_says_how_to_install_the_kernel(self):
        """Naming the missing package is not enough to act on.

        `agent-flow-core` is on no package index, so the reader's first move —
        `pip install agent-flow-core` — fails. A diagnostic that only reports
        absence leaves a server permanently missing a feature it ships, which
        is how this one stayed off. The answer has to carry the route back.
        """

        agent_id = self._store_agent(LINEAR_FLOW)

        with kernel_uninstalled():
            payload = self.client.get(f"/api/agent/agents/{agent_id}").json()

        unavailable = payload["flow_graph_unavailable"]
        message = unavailable["message"]
        # The declaration that names the dependency...
        self.assertIn("requirements-kernel.txt", message)
        # ...and the command that installs it.
        self.assertIn("sync-local-install.sh", message)
        self.assertIn("CODE_BRIDGE_FLOW_CORE_DIR", message)
        # Machine-readable too, so a client can show the command verbatim
        # instead of scraping an English sentence for it.
        self.assertIn("pip install", unavailable["remedy"])
        self.assertIn("requirements-kernel.txt", unavailable["remedy"])

    def test_only_the_kernel_absent_reason_offers_an_install(self):
        """The install hint answers "there is no kernel". Attaching it to a
        malformed workflow or a converter crash would send the reader to fix
        an environment that is already correct."""

        agent_id = self._store_agent(LINEAR_FLOW)

        def _explode(_steps):
            raise RuntimeError("converter blew up")

        with mock.patch("agent.flow_graph.to_graph", _explode):
            unavailable = self.client.get(
                f"/api/agent/agents/{agent_id}"
            ).json()["flow_graph_unavailable"]

        self.assertNotIn("remedy", unavailable)
        self.assertNotIn("requirements-kernel.txt", unavailable["message"])

    def test_the_rest_of_the_agent_is_untouched(self):
        agent_id = self._store_agent(LINEAR_FLOW)

        with kernel_uninstalled():
            payload = self.client.get(f"/api/agent/agents/{agent_id}").json()

        self.assertEqual(payload["id"], agent_id)
        self.assertEqual(payload["name"], "grapher")
        self.assertEqual(
            [step["id"] for step in payload["flow_json"]], ["plan", "tell"]
        )
        self.assertIn("origin", payload)
        self.assertIn("activation", payload)
        self.assertIn("next_fire_at", payload)

    def test_the_list_route_is_unaffected(self):
        """It never derived a graph, so a missing kernel changes nothing —
        including not adding an unavailable field it has no reason to carry."""

        self._store_agent(LINEAR_FLOW)

        with kernel_uninstalled():
            response = self.client.get("/api/agent/agents")

        self.assertEqual(response.status_code, 200, response.text)
        listed = response.json()["agents"]
        self.assertEqual(len(listed), 1)
        self.assertNotIn("flow_graph_unavailable", listed[0])


class UnconvertibleFlowTest(FlowGraphApiTestBase):
    """(c) A workflow that cannot become a graph says which one it is."""

    def test_dangling_goto_target_is_reported_not_swallowed(self):
        agent_id = self._store_agent(
            [
                {
                    "id": "plan",
                    "type": "llm",
                    "name": "Plan",
                    "instruction": "decide",
                    "on_failure": {
                        "type": "goto_step",
                        "target_step_id": "no_such_step",
                    },
                }
            ]
        )

        response = self.client.get(f"/api/agent/agents/{agent_id}")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertNotIn("flow_graph", payload)
        unavailable = payload["flow_graph_unavailable"]
        self.assertEqual(unavailable["reason"], "not_normalizable")
        # The reason must name the offending target, not just fail.
        self.assertIn("no_such_step", unavailable["message"])
        self.assertIn("no_such_step", unavailable["detail"])

    def test_a_reason_is_not_the_same_reason(self):
        """``not_normalizable`` and ``kernel_not_installed`` are different
        answers to different questions and must not collapse into one."""

        agent_id = self._store_agent([{"id": "plan", "type": "not_a_step_type"}])

        payload = self.client.get(f"/api/agent/agents/{agent_id}").json()

        self.assertEqual(
            payload["flow_graph_unavailable"]["reason"], "not_normalizable"
        )

    def test_unsupported_topology_passes_its_full_issue_list_through(self):
        """The converter's issue list is the useful part of its refusal.

        ``to_graph`` on a normalized list does not raise this today — the write
        path (T-B-05) and ``from_graph`` are where an arbitrary graph arrives —
        but the route must not flatten the issues to a string if it ever does.
        """

        from agent.flow_graph import UnsupportedTopologyError
        from agent_flow_core.validate import FlowIssue

        issue = FlowIssue(
            code="linear.edge_unbacked",
            severity="error",
            step_id="plan",
            edge_id="plan:success",
            message="edge is not derivable from any step policy",
            detail={"fromStepId": "plan", "toStepId": "tell"},
        )

        agent_id = self._store_agent(LINEAR_FLOW)

        def _refuse(_steps):
            raise UnsupportedTopologyError("graph is not foldable", issues=[issue])

        with mock.patch("agent.flow_graph.to_graph", _refuse):
            response = self.client.get(f"/api/agent/agents/{agent_id}")

        self.assertEqual(response.status_code, 200, response.text)
        unavailable = response.json()["flow_graph_unavailable"]
        self.assertEqual(unavailable["reason"], "not_linear")
        self.assertEqual(unavailable["message"], "graph is not foldable")
        self.assertEqual(
            [(item["code"], item["stepId"], item["edgeId"]) for item in unavailable["issues"]],
            [("linear.edge_unbacked", "plan", "plan:success")],
        )

    def test_an_unexpected_converter_error_is_reported_not_a_500(self):
        agent_id = self._store_agent(LINEAR_FLOW)

        def _explode(_steps):
            raise RuntimeError("converter blew up")

        with mock.patch("agent.flow_graph.to_graph", _explode):
            response = self.client.get(f"/api/agent/agents/{agent_id}")

        self.assertEqual(response.status_code, 200, response.text)
        unavailable = response.json()["flow_graph_unavailable"]
        self.assertEqual(unavailable["reason"], "conversion_failed")
        self.assertIn("RuntimeError", unavailable["detail"])


class DashboardMirrorTest(FlowGraphApiTestBase):
    """(d) The dashboard reads agents through the same handler.

    It delegates rather than reimplementing, which is what keeps the two
    surfaces identical — but only while it keeps delegating, and a mirror that
    silently stopped matching is how this project has produced 500s before.
    """

    def test_the_dashboard_gets_the_same_graph(self):
        agent_id = self._store_agent(LINEAR_FLOW)

        api = self.client.get(f"/api/agent/agents/{agent_id}").json()
        mirrored = self.dashboard.get(
            f"/api/dashboard/agent/agents/{agent_id}"
        ).json()

        self.assertEqual(mirrored["flow_graph"], api["flow_graph"])

    def test_the_dashboard_gets_the_same_reason_without_the_kernel(self):
        agent_id = self._store_agent(LINEAR_FLOW)

        with kernel_uninstalled():
            response = self.dashboard.get(
                f"/api/dashboard/agent/agents/{agent_id}"
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["flow_graph_unavailable"]["reason"],
            "kernel_not_installed",
        )


if __name__ == "__main__":
    unittest.main()


class KernelGateIssuesTest(unittest.TestCase):
    """The read publishes the kernel gate's verdict next to the graph.

    ``validate_flow`` has reported ``step.unreachable`` for a step with no
    way in since it was written; nothing in the product called it, so the
    morning-check agent's ``analyze_failure`` drew as a second root for weeks
    with the kernel ready to say so on the first read.
    """

    def _view(self, flow):
        return agents._flow_graph_view(flow)

    def test_a_clean_graph_publishes_an_empty_list_not_no_key(self):
        view = self._view(LINEAR_FLOW)
        self.assertIn("flow_graph", view)
        self.assertEqual(view["flow_graph_issues"], [])

    def test_a_step_nothing_leads_to_is_named(self):
        # `tell` ends the run; nothing routes into `after`. The graph still
        # draws — `after` is simply a second root — and the gate says so.
        # (Three steps, not two: the gate skips an edgeless flow, which by
        # convention runs in list order.)
        flow = [
            LINEAR_FLOW[0],
            {**LINEAR_FLOW[1], "on_success": {"type": "end"}},
            {**LINEAR_FLOW[1], "id": "after", "name": "After"},
        ]
        view = self._view(flow)
        self.assertIn("flow_graph", view)
        self.assertEqual(
            [(i["code"], i["severity"], i.get("stepId")) for i in view["flow_graph_issues"]],
            [("step.unreachable", "warning", "after")],
        )

    def test_the_canvas_graph_view_carries_it_too(self):
        from routes import canvas_api

        self.assertIn("flow_graph_issues", canvas_api._GRAPH_VIEW_KEYS)
