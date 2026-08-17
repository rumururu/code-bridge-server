"""The write API's graph input (agent-flow-core T-B-05).

``POST /agents`` and ``PATCH /agents/{id}`` accept ``flow_graph``: the same
kernel wire-form ``Flow`` dict that ``GET /agents/{id}`` serves. The route
folds it through ``agent.flow_graph.from_graph`` into the linear ``flow_json``
before anything is stored — **the stored canon stays linear**; the graph is an
input representation, never a second source of truth. Pinned here:

1. **Round trip.** A graph read from one agent creates another whose stored
   ``flow_json`` is identical, and whose re-read ``flow_graph`` is identical.

2. **No silent preference.** A body carrying both ``flow_json`` and
   ``flow_graph`` is ambiguous about which is meant and is refused (400),
   never resolved by quietly picking one.

3. **Refusals carry the issues.** A graph outside the linear subset is a 400
   with ``from_graph``'s full ``linear.*`` issue list verbatim — the caller
   sees what to fix, not just that folding failed.

4. **A missing kernel refuses the write out loud.** The read view degrades to
   ``flow_graph_unavailable`` when ``agent_flow_core`` is absent; a write
   cannot degrade — dropping the caller's graph would lose their work — so it
   answers 422 with ``reason: kernel_not_installed``, and plain ``flow_json``
   writes keep working untouched.

5. **Same gate, different door.** A workflow arriving as a graph faces the
   same commit contract gate as one arriving as ``flow_json``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from agent.browser_action_adapter import reset_browser_readiness_cache  # noqa: E402
from agent.workflow_v2 import normalize_workflow  # noqa: E402
from core import database  # noqa: E402
from routes import agents, dashboard_agents  # noqa: E402
from routes.deps import require_local_access, verify_api_key  # noqa: E402
from test_flow_graph_api import LINEAR_FLOW, kernel_uninstalled  # noqa: E402


def wire_graph(flow_json: list[dict]) -> dict:
    """The kernel wire form of a linear workflow — what a graph client sends.

    Built through the same converter the read view uses, so these tests
    exercise exactly the representation ``GET /agents/{id}`` hands out.
    (This helper needs the kernel; the kernel-absent tests never call it
    inside the simulation.)
    """

    from agent.flow_graph import to_graph

    return to_graph(normalize_workflow(flow_json)).model_dump(by_alias=True)


SECOND_FLOW = [
    {"id": "ping", "type": "llm", "name": "Ping", "instruction": "say ping"},
]

BROWSER_PLACEHOLDER_FLOW = [
    {
        "id": "open_cafe",
        "name": "카페 열기",
        "type": "browser_action",
        "description": "카페 글쓰기 페이지를 연다.",
        "actions": [
            {"type": "navigate", "url": "configured_cafe_url"},
            {"type": "click", "selector": "#write"},
        ],
    }
]

READY_RUNTIME = {
    "ready": True,
    "playwright_python": True,
    "chromium_executable": True,
    "install_command": "/opt/venv/bin/python -m playwright install chromium",
    "message": "",
}

AGENT_BODY = {
    "name": "grapher",
    "description": "flow graph write fixture",
    "system_prompt": "You are useful.",
    "provider_id": "openai",
}


class FlowGraphWriteApiTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_flow_graph_write.db"
        agent_store._agent_store = None
        schedule_store._store = None
        reset_browser_readiness_cache()
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
        reset_browser_readiness_cache()
        self._tmp.cleanup()

    # --- helpers ---------------------------------------------------------

    def _create(self, client=None, **fields) -> dict:
        response = (client or self.client).post(
            "/api/agent/agents", json={**AGENT_BODY, **fields}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _stored_flow(self, agent_id: str) -> list[dict]:
        return self.store.get_agent(agent_id)["flow_json"]

    def _readiness(self, snapshot):
        """Pin the browser-readiness answer, exactly as the commit-gate tests
        do — leaving the real probes in would let *this* machine's Playwright
        install decide the assertion."""
        return mock.patch.multiple(
            agents,
            get_cached_browser_readiness_sync=mock.Mock(return_value=snapshot),
            get_browser_runtime_readiness=mock.AsyncMock(return_value=snapshot),
        )


class GraphCreateTest(FlowGraphWriteApiTestBase):
    """(a) Creating from a graph stores the same linear canon."""

    def test_create_from_graph_stores_the_folded_linear_flow(self):
        created = self._create(flow_graph=wire_graph(LINEAR_FLOW))

        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(LINEAR_FLOW)
        )
        # The response echoes what was saved — a linear flow, not a graph.
        self.assertEqual(
            [step["id"] for step in created["flow_json"]], ["plan", "tell"]
        )

    def test_the_read_graph_round_trips_through_a_create(self):
        """GET one agent's graph, create a second from it, GET that: all three
        agree — the wire form the read serves is the wire form the write eats."""

        via_json = self._create(flow_json=LINEAR_FLOW)
        source = self.client.get(f"/api/agent/agents/{via_json['id']}").json()

        via_graph = self._create(name="grapher2", flow_graph=source["flow_graph"])
        echoed = self.client.get(f"/api/agent/agents/{via_graph['id']}").json()

        self.assertEqual(
            self._stored_flow(via_graph["id"]), self._stored_flow(via_json["id"])
        )
        self.assertNotIn("flow_graph_unavailable", echoed)
        self.assertEqual(echoed["flow_graph"], source["flow_graph"])


class GraphUpdateTest(FlowGraphWriteApiTestBase):
    """(b) Updating with a graph replaces the linear canon the same way."""

    def test_update_from_graph_stores_the_folded_linear_flow(self):
        created = self._create(flow_json=LINEAR_FLOW)

        response = self.client.patch(
            f"/api/agent/agents/{created['id']}",
            json={"flow_graph": wire_graph(SECOND_FLOW)},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(SECOND_FLOW)
        )
        # And the read view now derives the new graph — round trip after update.
        echoed = self.client.get(f"/api/agent/agents/{created['id']}").json()
        self.assertEqual(echoed["flow_graph"], wire_graph(SECOND_FLOW))


class BothRepresentationsTest(FlowGraphWriteApiTestBase):
    """(c) flow_json + flow_graph in one body is ambiguous and refused."""

    def test_create_with_both_is_a_400_and_writes_nothing(self):
        response = self.client.post(
            "/api/agent/agents",
            json={
                **AGENT_BODY,
                "flow_json": LINEAR_FLOW,
                "flow_graph": wire_graph(LINEAR_FLOW),
            },
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"], "flow_input_conflict")
        self.assertEqual(self.store.count_agents(), 0)

    def test_update_with_both_is_a_400_and_changes_nothing(self):
        created = self._create(flow_json=LINEAR_FLOW)

        response = self.client.patch(
            f"/api/agent/agents/{created['id']}",
            json={
                "flow_json": SECOND_FLOW,
                "flow_graph": wire_graph(SECOND_FLOW),
            },
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"], "flow_input_conflict")
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(LINEAR_FLOW)
        )

    def test_an_explicitly_null_flow_graph_is_not_a_conflict(self):
        """``flow_graph: null`` says "no graph input", which is what omitting
        the field says — it must not poison a plain flow_json write."""

        created = self._create(flow_json=LINEAR_FLOW)

        response = self.client.patch(
            f"/api/agent/agents/{created['id']}",
            json={"flow_json": SECOND_FLOW, "flow_graph": None},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(SECOND_FLOW)
        )


class NonLinearGraphTest(FlowGraphWriteApiTestBase):
    """(d) A graph outside the linear subset is refused with the issue list."""

    def _nonlinear(self) -> dict:
        graph = wire_graph(LINEAR_FLOW)
        graph = deepcopy(graph)
        # A back-edge no step policy derives — from_graph's linear.edge_unbacked.
        graph["edges"].append(
            {
                "id": "tell:success",
                "fromStepId": "tell",
                "toStepId": "plan",
                "kind": "goto",
            }
        )
        return graph

    def test_create_is_refused_with_the_linear_issue_list(self):
        response = self.client.post(
            "/api/agent/agents", json={**AGENT_BODY, "flow_graph": self._nonlinear()}
        )

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        self.assertEqual(payload["error"], "unsupported_topology")
        issues = payload["issues"]
        self.assertTrue(issues, payload)
        self.assertEqual(
            [(item["code"], item["edgeId"]) for item in issues],
            [("linear.edge_unbacked", "tell:success")],
        )
        # The summary sentence names the offender too.
        self.assertIn("tell:success", payload["detail"])
        self.assertEqual(self.store.count_agents(), 0)

    def test_update_is_refused_the_same_way_and_changes_nothing(self):
        created = self._create(flow_json=LINEAR_FLOW)

        response = self.client.patch(
            f"/api/agent/agents/{created['id']}",
            json={"flow_graph": self._nonlinear()},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"], "unsupported_topology")
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(LINEAR_FLOW)
        )

    def test_a_dict_that_is_not_a_flow_at_all_is_a_400_not_a_500(self):
        response = self.client.post(
            "/api/agent/agents",
            json={**AGENT_BODY, "flow_graph": {"nodes": "nope"}},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"], "invalid_flow_graph")
        self.assertEqual(self.store.count_agents(), 0)


class KernelMissingWriteTest(FlowGraphWriteApiTestBase):
    """(e) No kernel: graph writes refuse out loud; flow_json writes work on."""

    def test_create_with_graph_is_a_422_naming_the_kernel(self):
        graph = wire_graph(LINEAR_FLOW)  # built before the simulation

        with kernel_uninstalled():
            response = self.client.post(
                "/api/agent/agents", json={**AGENT_BODY, "flow_graph": graph}
            )

        self.assertEqual(response.status_code, 422, response.text)
        payload = response.json()
        self.assertEqual(payload["reason"], "kernel_not_installed")
        self.assertEqual(payload["error"], "kernel_not_installed")
        self.assertIn("agent-flow-core", payload["detail"])
        # Refused means refused: nothing half-saved.
        self.assertEqual(self.store.count_agents(), 0)

    def test_update_with_graph_is_a_422_and_changes_nothing(self):
        created = self._create(flow_json=LINEAR_FLOW)
        graph = wire_graph(SECOND_FLOW)

        with kernel_uninstalled():
            response = self.client.patch(
                f"/api/agent/agents/{created['id']}", json={"flow_graph": graph}
            )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["reason"], "kernel_not_installed")
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(LINEAR_FLOW)
        )

    def test_flow_json_writes_do_not_need_the_kernel(self):
        """The linear canon predates the kernel and must outlive its absence —
        a deployed venv without agent-flow-core keeps full write service."""

        with kernel_uninstalled():
            created = self._create(flow_json=LINEAR_FLOW)
            patched = self.client.patch(
                f"/api/agent/agents/{created['id']}",
                json={"flow_json": SECOND_FLOW},
            )

        self.assertEqual(patched.status_code, 200, patched.text)
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(SECOND_FLOW)
        )


class DashboardMirrorWriteTest(FlowGraphWriteApiTestBase):
    """(f) The dashboard writes agents through the same handlers and body
    models (``dashboard_agents`` imports ``AgentCreateBody``/``AgentUpdateBody``
    from ``routes.agents``), so graph input must behave identically there —
    a mirror that silently stopped matching is how this project has produced
    500s before."""

    def test_dashboard_create_from_graph_matches_the_api(self):
        via_api = self._create(flow_graph=wire_graph(LINEAR_FLOW))

        response = self.dashboard.post(
            "/api/dashboard/agent/agents",
            json={**AGENT_BODY, "name": "mirrored", "flow_graph": wire_graph(LINEAR_FLOW)},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self._stored_flow(response.json()["id"]),
            self._stored_flow(via_api["id"]),
        )

    def test_dashboard_update_from_graph_matches_the_api(self):
        created = self._create(flow_json=LINEAR_FLOW)

        response = self.dashboard.patch(
            f"/api/dashboard/agent/agents/{created['id']}",
            json={"flow_graph": wire_graph(SECOND_FLOW)},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(SECOND_FLOW)
        )

    def test_dashboard_refuses_both_representations_identically(self):
        response = self.dashboard.post(
            "/api/dashboard/agent/agents",
            json={
                **AGENT_BODY,
                "flow_json": LINEAR_FLOW,
                "flow_graph": wire_graph(LINEAR_FLOW),
            },
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"], "flow_input_conflict")


class ContractGateTest(FlowGraphWriteApiTestBase):
    """(g) Graph input faces the same commit gate as flow_json input."""

    def test_a_stalling_workflow_is_refused_no_matter_which_shape_it_wore(self):
        graph = wire_graph(BROWSER_PLACEHOLDER_FLOW)

        with self._readiness(READY_RUNTIME):
            as_graph = self.client.post(
                "/api/agent/agents", json={**AGENT_BODY, "flow_graph": graph}
            )
            as_json = self.client.post(
                "/api/agent/agents",
                json={**AGENT_BODY, "flow_json": BROWSER_PLACEHOLDER_FLOW},
            )

        self.assertEqual(as_graph.status_code, 400, as_graph.text)
        self.assertEqual(as_json.status_code, 400, as_json.text)
        # Not merely both refused — the *same* refusal, from the same gate.
        self.assertEqual(
            as_graph.json()["error"], as_json.json()["error"]
        )
        self.assertEqual(as_graph.json()["error"], "unresolved_browser_targets")
        self.assertEqual(
            as_graph.json()["unresolved"], as_json.json()["unresolved"]
        )
        self.assertEqual(self.store.count_agents(), 0)

    def test_commit_incomplete_is_honored_for_graph_input_too(self):
        """The gate's escape hatch belongs to the gate, not to the input
        shape: saving deliberately incomplete works, and says so."""

        with self._readiness(READY_RUNTIME):
            response = self.client.post(
                "/api/agent/agents",
                json={
                    **AGENT_BODY,
                    "flow_graph": wire_graph(BROWSER_PLACEHOLDER_FLOW),
                    "commit_incomplete": True,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["readiness"]["ok"])
        self.assertEqual(
            self._stored_flow(payload["id"]),
            normalize_workflow(BROWSER_PLACEHOLDER_FLOW),
        )

    def test_update_with_graph_passes_the_same_gate(self):
        created = self._create(flow_json=LINEAR_FLOW)

        with self._readiness(READY_RUNTIME):
            response = self.client.patch(
                f"/api/agent/agents/{created['id']}",
                json={"flow_graph": wire_graph(BROWSER_PLACEHOLDER_FLOW)},
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"], "unresolved_browser_targets")
        self.assertEqual(
            self._stored_flow(created["id"]), normalize_workflow(LINEAR_FLOW)
        )


if __name__ == "__main__":
    unittest.main()
