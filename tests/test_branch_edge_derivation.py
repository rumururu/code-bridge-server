"""A branching workflow is drawn as branching — both places it is derived.

Track H wave 2 (T-H-08 / T-H-09 / T-H-10). Before this, a ``condition`` step
with two arms came out of *both* derivation points as a straight line:

    run_sync -> check_exit  [seq]
    check_exit -> diagnose  [seq]     <- claims the condition always goes here
    from_graph(to_graph(L)) == L  ->  True

The round trip passed because ``to_graph`` and ``from_graph`` were blind to
``branches`` in exactly the same way — a blind fold agrees with a blind
unfold. **So a round-trip proposition can never catch this class of error**,
and none of the assertions below rest on one: the tests state the ``kind``
and the *count* of the edges leaving a branch node directly
(RUNNER_BRANCHING_SPEC 7.2 asks for precisely that). The round trip is still
checked, as a necessary-not-sufficient condition.

The three worked examples are not retyped here. They are **read out of
``docs/concept/spec/RUNNER_BRANCHING_SPEC.md`` section 10 at import time** —
the flow_json and the expected edge list, both — so the spec cannot drift
away from the code without this file failing. If an example stops surviving
the derivation, the answer is to fix whichever of the two is wrong, in the
open, rather than to quietly edit the example until it matches the code.

The two derivation points are separate code with one shared rule
(LINEAR_FLOW_MAPPING 3.4): ``flow_graph.to_graph`` reads normalized *steps*
and keys nodes by step id, ``step_cursor.derive_run_graph`` reads a run's
*rows* and keys nodes by position. Both are exercised against the same three
examples here, because "E5 replaces E1/E2" landing in only one of them is the
failure this track has already seen once (a canvas that draws arms over a run
graph that has none).
"""

from __future__ import annotations

import copy
import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any

SERVER_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for _path in (str(SERVER_DIR), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.flow_graph import (  # noqa: E402
    UnsupportedTopologyError,
    from_graph,
    to_graph,
)
from code_bridge_core.step_cursor import (  # noqa: E402
    CONTROL_EDGE_KINDS,
    UNSUPPORTED_TOPOLOGY_REASON,
    StepCursor,
    derive_run_graph,
    plan_execution_order,
)
from agent import task_orchestrator  # noqa: E402
from agent.task_orchestrator import _plan_workflow_steps  # noqa: E402
from code_bridge_core.workflow_v2 import ALLOWED_STEP_TYPES, normalize_workflow  # noqa: E402
from agent_flow_core.errors import CyclicFlowError  # noqa: E402
from agent_flow_core.model import Flow  # noqa: E402
from agent_flow_core.topology import (  # noqa: E402
    CONTROL_EDGE_KINDS as KERNEL_CONTROL_EDGE_KINDS,
    topological_sort,
)
from agent_flow_core.validate import validate_flow  # noqa: E402

SPEC_PATH = (
    SERVER_DIR.parent / "docs" / "concept" / "spec" / "RUNNER_BRANCHING_SPEC.md"
)


def _load_spec_examples() -> dict[str, tuple[list[dict], list[dict]]]:
    """``{"A": (flow_json, expected_edges), ...}`` from the spec's section 10.

    Each example is a ``### 10.N 예시 X`` heading followed by two ```json
    blocks: the normalized flow_json, then the edges it derives.
    """

    text = SPEC_PATH.read_text(encoding="utf-8")
    parts = re.split(r"^### 10\.\d+ 예시 ([ABC])", text, flags=re.M)
    examples: dict[str, tuple[list[dict], list[dict]]] = {}
    for index in range(1, len(parts), 2):
        letter, body = parts[index], parts[index + 1]
        blocks = re.findall(r"```json\n(.*?)```", body, flags=re.S)
        examples[letter] = (json.loads(blocks[0]), json.loads(blocks[1]))
    return examples


EXAMPLES = _load_spec_examples()

#: The step whose branches each example is about, and the execution order the
#: spec records for it (prose lines "topological_sort 결과: ...").
BRANCH_STEP = {"A": "check_exit", "B": "route", "C": "still_running"}
SPEC_ORDER = {
    "A": ["run_sync", "check_exit", "diagnose", "notify_ok"],
    "B": ["read_stock", "route", "restock", "warn", "notify_ok"],
    "C": ["poll", "still_running", "report"],
}


def _rows(flow_json: list[dict]) -> list[dict[str, Any]]:
    """The run step rows a run would be planned with for this flow_json."""

    rows = _plan_workflow_steps(normalize_workflow(copy.deepcopy(flow_json)), [])
    for index, row in enumerate(rows):
        row["id"] = f"row-{index}"
    return rows


def _wire(edge: Any) -> dict[str, Any]:
    """An edge in the shape the spec writes it (no data-binding fields)."""

    return {
        "id": edge.id,
        "fromStepId": edge.from_step_id,
        "toStepId": edge.to_step_id,
        "kind": edge.kind,
        "extensions": edge.extensions,
    }


def _issue_codes(error: UnsupportedTopologyError) -> list[str]:
    return sorted(issue.code for issue in error.issues)


class SpecExamplesAreReallyLoadedTest(unittest.TestCase):
    """Nothing below is allowed to pass because the spec failed to parse."""

    def test_all_three_examples_were_read(self) -> None:
        self.assertEqual(sorted(EXAMPLES), ["A", "B", "C"])

    def test_each_example_has_a_branching_condition_and_expected_edges(self) -> None:
        expected_arm_counts = {"A": 2, "B": 3, "C": 2}
        for letter, (flow_json, edges) in EXAMPLES.items():
            with self.subTest(example=letter):
                step = next(
                    s for s in flow_json if s["id"] == BRANCH_STEP[letter]
                )
                self.assertEqual(step["type"], "condition")
                self.assertEqual(
                    len(step["branches"]), expected_arm_counts[letter]
                )
                self.assertEqual(
                    len([e for e in edges if e["kind"] == "branch"]),
                    expected_arm_counts[letter],
                )

    def test_the_examples_are_already_normalized(self) -> None:
        # The spec says its flow_json blocks are `normalize_workflow` output.
        # If that stops being true the examples are not the shape the runner
        # would actually store, and every derivation below is testing fiction.
        for letter, (flow_json, _) in EXAMPLES.items():
            with self.subTest(example=letter):
                self.assertEqual(
                    normalize_workflow(copy.deepcopy(flow_json)), flow_json
                )


class AuthoringDerivationTest(unittest.TestCase):
    """T-H-10 — ``flow_graph.to_graph`` draws the arms the spec draws."""

    def test_to_graph_matches_the_spec_edges_exactly(self) -> None:
        for letter, (flow_json, expected) in EXAMPLES.items():
            with self.subTest(example=letter):
                graph = to_graph(normalize_workflow(copy.deepcopy(flow_json)))
                self.assertEqual([_wire(edge) for edge in graph.edges], expected)
                self.assertTrue(
                    all(
                        edge.from_field is None and edge.to_field is None
                        for edge in graph.edges
                    )
                )

    def test_a_branching_condition_draws_arms_and_no_sequential_edge(self) -> None:
        # The direct assertion RUNNER_BRANCHING_SPEC 7.2 asks for: kind and
        # count of the edges leaving the branch node, not a round-trip bool.
        for letter, (flow_json, _) in EXAMPLES.items():
            with self.subTest(example=letter):
                steps = normalize_workflow(copy.deepcopy(flow_json))
                branch_step = BRANCH_STEP[letter]
                arms = next(s for s in steps if s["id"] == branch_step)["branches"]
                outgoing = [
                    edge
                    for edge in to_graph(steps).edges
                    if edge.from_step_id == branch_step
                ]
                self.assertEqual(len(outgoing), len(arms))
                self.assertEqual(
                    [edge.kind for edge in outgoing], ["branch"] * len(arms)
                )
                self.assertEqual(
                    [edge.to_step_id for edge in outgoing],
                    [arm["target_step_id"] for arm in arms],
                )

    def test_the_arm_annotation_carries_label_and_default_but_no_predicate(
        self,
    ) -> None:
        steps = normalize_workflow(copy.deepcopy(EXAMPLES["B"][0]))
        outgoing = [e for e in to_graph(steps).edges if e.from_step_id == "route"]
        self.assertEqual(
            [e.extensions["codeBridgeBranch"] for e in outgoing],
            [
                {"index": 0, "label": "품절", "default": False},
                {"index": 1, "label": "소량", "default": False},
                {"index": 2, "label": "충분", "default": True},
            ],
        )
        # No dual representation: the predicate lives on the node only.
        for edge in outgoing:
            self.assertNotIn("when", edge.extensions["codeBridgeBranch"])
            self.assertEqual(edge.extensions["codeBridgeLinear"], {"on": "success"})

    def test_round_trip_holds_for_all_three(self) -> None:
        # Necessary, not sufficient — see this module's docstring. It only
        # says the two directions agree, which they also did while both were
        # wrong.
        for letter, (flow_json, _) in EXAMPLES.items():
            with self.subTest(example=letter):
                linear = normalize_workflow(copy.deepcopy(flow_json))
                self.assertEqual(from_graph(to_graph(linear)), linear)

    def test_the_graph_round_trips_the_other_way_too(self) -> None:
        for letter, (flow_json, _) in EXAMPLES.items():
            with self.subTest(example=letter):
                graph = to_graph(normalize_workflow(copy.deepcopy(flow_json)))
                again = to_graph(from_graph(graph))
                self.assertEqual(
                    again.model_dump(by_alias=True), graph.model_dump(by_alias=True)
                )

    def test_the_kernel_gate_accepts_the_branching_graphs(self) -> None:
        for letter, (flow_json, _) in EXAMPLES.items():
            with self.subTest(example=letter):
                graph = to_graph(normalize_workflow(copy.deepcopy(flow_json)))
                issues = validate_flow(graph, allowed_step_types=ALLOWED_STEP_TYPES)
                self.assertEqual([i for i in issues if i.severity == "error"], [])

    def test_the_kernel_orders_each_example_as_the_spec_records(self) -> None:
        for letter, (flow_json, edges) in EXAMPLES.items():
            with self.subTest(example=letter):
                nodes = [{"id": step["id"]} for step in flow_json]
                ordered = topological_sort(nodes, edges)
                self.assertEqual([n["id"] for n in ordered], SPEC_ORDER[letter])

    def test_the_backward_arm_of_the_polling_loop_is_not_a_cycle(self) -> None:
        # Example C only sorts because `kind="branch"` is excluded from
        # topology (T-H-02). Drawn as `seq` the same graph is a cycle — the
        # spec records that comparison, so it is run here rather than quoted.
        _, edges = EXAMPLES["C"]
        nodes = [{"id": step["id"]} for step in EXAMPLES["C"][0]]
        as_seq = copy.deepcopy(edges)
        for edge in as_seq:
            if edge["kind"] == "branch":
                edge["kind"] = "seq"
        with self.assertRaises(CyclicFlowError):
            topological_sort(nodes, as_seq)


class RunDerivationTest(unittest.TestCase):
    """T-H-09 — ``step_cursor.derive_run_graph`` sees the same arms."""

    def _graph(self, letter: str):
        rows = _rows(EXAMPLES[letter][0])
        nodes, edges = derive_run_graph(rows)
        by_position = {node["id"]: node["workflowStepId"] for node in nodes}
        named = [
            (by_position[e["fromStepId"]], by_position[e["toStepId"]], e["kind"])
            for e in edges
        ]
        return rows, nodes, edges, named

    def test_the_run_rows_carry_the_branches(self) -> None:
        # Without this the derivation below finds no arms however right it
        # is (RUNNER_BRANCHING_SPEC 9.2) — the gap T-H-09 had to close first.
        rows = _rows(EXAMPLES["A"][0])
        self.assertEqual(
            rows[1]["input"]["branches"],
            next(
                step
                for step in EXAMPLES["A"][0]
                if step["id"] == "check_exit"
            )["branches"],
        )

    def test_each_example_derives_one_branch_edge_per_arm(self) -> None:
        for letter in ("A", "B", "C"):
            with self.subTest(example=letter):
                flow_json, _ = EXAMPLES[letter]
                arms = next(
                    s for s in flow_json if s["id"] == BRANCH_STEP[letter]
                )["branches"]
                _, _, _, named = self._graph(letter)
                outgoing = [
                    entry for entry in named if entry[0] == BRANCH_STEP[letter]
                ]
                self.assertEqual(
                    outgoing,
                    [
                        (BRANCH_STEP[letter], arm["target_step_id"], "branch")
                        for arm in arms
                    ],
                )

    def test_the_two_derivation_points_agree_on_kinds_and_counts(self) -> None:
        # Same rule, two implementations (LINEAR_FLOW_MAPPING 3.4). Compared
        # as (from, to, kind) triples because node identity differs: step ids
        # in one, row positions in the other.
        for letter in ("A", "B", "C"):
            with self.subTest(example=letter):
                steps = normalize_workflow(copy.deepcopy(EXAMPLES[letter][0]))
                authoring = [
                    (e.from_step_id, e.to_step_id, e.kind)
                    for e in to_graph(steps).edges
                ]
                _, _, _, run_time = self._graph(letter)
                self.assertEqual(sorted(run_time), sorted(authoring))

    def test_a_branching_condition_row_draws_no_sequential_successor(self) -> None:
        for letter in ("A", "B", "C"):
            with self.subTest(example=letter):
                _, _, _, named = self._graph(letter)
                self.assertEqual(
                    [
                        entry
                        for entry in named
                        if entry[0] == BRANCH_STEP[letter] and entry[2] == "seq"
                    ],
                    [],
                )

    def test_the_kernel_still_orders_the_run_in_list_order(self) -> None:
        for letter in ("A", "B", "C"):
            with self.subTest(example=letter):
                rows, nodes, edges, _ = self._graph(letter)
                order = plan_execution_order(nodes, edges)
                self.assertEqual(order.source, "kernel")
                self.assertIsNone(order.unsupported)
                self.assertEqual(order.positions, tuple(range(len(rows))))

    def test_a_condition_without_branches_still_sequences(self) -> None:
        # The pre-branching shape, unchanged: no `branches` key, so E1.
        rows = _rows(
            [
                {"id": "check", "type": "condition", "name": "Check"},
                {"id": "after", "type": "notify", "name": "After"},
            ]
        )
        _, edges = derive_run_graph(rows)
        self.assertEqual(
            edges,
            [{"fromStepId": "0", "toStepId": "1", "kind": "seq", "on": "success"}],
        )

    def test_an_empty_branches_list_still_sequences(self) -> None:
        # A half-authored condition node has no arms to draw, so it keeps the
        # sequential successor (spec 1.5); the authoring gate is what refuses
        # to save it.
        rows = _rows(
            [
                {
                    "id": "check",
                    "type": "condition",
                    "name": "Check",
                    "branches": [],
                },
                {"id": "after", "type": "notify", "name": "After"},
            ]
        )
        _, edges = derive_run_graph(rows)
        self.assertEqual([edge["kind"] for edge in edges], ["seq"])

    def test_an_arm_naming_no_row_draws_no_edge(self) -> None:
        # Same rule goto already had: an unresolvable target draws nothing
        # and routing aborts by name at run time.
        rows = _rows(EXAMPLES["A"][0])
        rows[1]["input"]["branches"][0]["target_step_id"] = "nowhere"
        _, edges = derive_run_graph(rows)
        branch_edges = [edge for edge in edges if edge["kind"] == "branch"]
        self.assertEqual(len(branch_edges), 1)
        self.assertEqual(branch_edges[0]["toStepId"], "3")  # notify_ok


class BranchIsNotAFanOutTest(unittest.TestCase):
    """T-H-08 — the park defence stops mistaking arms for parallel branches."""

    def test_a_two_armed_condition_does_not_park(self) -> None:
        rows = _rows(EXAMPLES["A"][0])
        route = StepCursor(rows).advance_on_success(rows, 1)
        self.assertNotEqual(route.kind, "park")
        self.assertIsNone(
            plan_execution_order(*derive_run_graph(rows)).unsupported
        )

    def test_a_three_armed_condition_does_not_park_either(self) -> None:
        rows = _rows(EXAMPLES["B"][0])
        self.assertIsNone(
            plan_execution_order(*derive_run_graph(rows)).unsupported
        )
        self.assertNotEqual(
            StepCursor(rows).advance_on_success(rows, 1).kind, "park"
        )

    def test_arms_converging_on_one_step_are_not_a_merge(self) -> None:
        # Two arms of one condition can name the same destination. That is
        # one path taken twice over, never two paths joining, so the merge
        # refusal must not read it as one.
        rows = _rows(
            [
                {"id": "poll", "type": "shell", "name": "Poll", "script_id": "s1"},
                {
                    "id": "route",
                    "type": "condition",
                    "name": "Route",
                    "branches": [
                        {
                            "label": "a",
                            "when": {
                                "left": "{{x}}",
                                "op": "equals",
                                "right": "1",
                            },
                            "target_step_id": "done",
                        },
                        {"label": "b", "when": None, "target_step_id": "done"},
                    ],
                },
                {"id": "done", "type": "notify", "name": "Done"},
            ]
        )
        order = plan_execution_order(*derive_run_graph(rows))
        self.assertIsNone(order.unsupported)

    def test_the_control_kinds_match_the_kernels(self) -> None:
        # `step_cursor` keeps its own literal rather than importing this one,
        # because the runner's hot path must not depend on the kernel being
        # installed (see `_kernel_topology`). A copy can drift, so the copy
        # is compared here, where the kernel is present.
        self.assertEqual(CONTROL_EDGE_KINDS, KERNEL_CONTROL_EDGE_KINDS)

    def test_a_sequential_fan_out_is_still_refused(self) -> None:
        # The stage-2 line: parallel execution stays refused. Widening the
        # defence to "any two successors are fine" is the silent
        # linearization T-B-07 exists to prevent.
        rows = _rows(
            [
                {"id": "fan", "type": "llm", "name": "Fan"},
                {"id": "left", "type": "llm", "name": "Left"},
                {"id": "right", "type": "llm", "name": "Right"},
            ]
        )
        rows[0]["input"]["on_success"] = {
            "type": "parallel",
            "targets": ["left", "right"],
        }
        order = plan_execution_order(*derive_run_graph(rows))
        self.assertIsNotNone(order.unsupported)
        self.assertEqual(
            StepCursor(rows).advance_on_success(rows, 0).park_reason,
            UNSUPPORTED_TOPOLOGY_REASON,
        )

    def test_a_sequential_merge_is_still_refused(self) -> None:
        rows = _rows(
            [
                {"id": "left", "type": "llm", "name": "Left"},
                {"id": "right", "type": "llm", "name": "Right"},
                {"id": "join", "type": "llm", "name": "Join"},
            ]
        )
        rows[0]["input"]["on_success"] = {"type": "parallel", "targets": ["join"]}
        order = plan_execution_order(*derive_run_graph(rows))
        self.assertIsNotNone(order.unsupported)
        self.assertIn("join back into", order.unsupported)

    def test_the_park_guidance_points_at_condition_branches(self) -> None:
        rows = _rows(
            [
                {"id": "fan", "type": "llm", "name": "Fan"},
                {"id": "left", "type": "llm", "name": "Left"},
                {"id": "right", "type": "llm", "name": "Right"},
            ]
        )
        rows[0]["input"]["on_success"] = {
            "type": "parallel",
            "targets": ["left", "right"],
        }
        prompt = plan_execution_order(*derive_run_graph(rows)).unsupported or ""
        self.assertIn("condition step", prompt)
        self.assertIn("one step at a time", prompt)

    def test_the_korean_guidance_says_the_same_thing(self) -> None:
        # Two languages, one instruction. The English prompt above is the
        # cursor's; this is the checkpoint's, and they drifted apart in the
        # ticket that first wrote them.
        action = task_orchestrator._required_user_action(UNSUPPORTED_TOPOLOGY_REASON)
        self.assertIn("condition", action)
        self.assertIn("분기", action)
        self.assertNotIn("한 줄로", action)


class BranchAcceptanceTest(unittest.TestCase):
    """T-H-10 — what ``from_graph`` accepts and refuses, by issue code."""

    def _graph_for(self, letter: str) -> Flow:
        return to_graph(normalize_workflow(copy.deepcopy(EXAMPLES[letter][0])))

    def _rejected(self, flow: Flow) -> UnsupportedTopologyError:
        with self.assertRaises(UnsupportedTopologyError) as ctx:
            from_graph(flow)
        return ctx.exception

    def test_a_default_branch_drawn_first_still_loads(self) -> None:
        # The canvas may draw the default arm anywhere; normalization moves
        # it last (spec 1.4). So the annotation's `index` legitimately
        # disagrees with the re-derived one and is advisory — only `label`
        # and `default` are compared.
        flow_json = copy.deepcopy(EXAMPLES["B"][0])
        route = next(s for s in flow_json if s["id"] == "route")
        route["branches"] = [route["branches"][2]] + route["branches"][:2]
        graph = to_graph(
            normalize_workflow(copy.deepcopy(flow_json))
        )  # normalization already reordered
        # Re-label the edges the way a canvas that drew the default first
        # would: default annotated as index 0.
        drawn = graph.model_dump(by_alias=True)
        for edge in drawn["edges"]:
            annotation = edge["extensions"].get("codeBridgeBranch")
            if annotation is not None:
                annotation["index"] = (annotation["index"] + 1) % 3
        restored = from_graph(Flow.model_validate(drawn))
        self.assertEqual(
            [arm["label"] for arm in restored[1]["branches"]],
            ["품절", "소량", "충분"],
        )

    def test_an_arm_label_that_contradicts_the_node_is_rejected(self) -> None:
        drawn = self._graph_for("A").model_dump(by_alias=True)
        drawn["edges"][1]["extensions"]["codeBridgeBranch"]["label"] = "성공"
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("linear.edge_meta_mismatch", _issue_codes(error))

    def test_a_branch_annotation_on_a_sequential_edge_is_rejected(self) -> None:
        drawn = self._graph_for("A").model_dump(by_alias=True)
        drawn["edges"][0]["extensions"]["codeBridgeBranch"] = {
            "index": 0,
            "label": "x",
            "default": False,
        }
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("linear.edge_meta_mismatch", _issue_codes(error))

    def test_a_branch_edge_from_a_non_condition_step_is_rejected(self) -> None:
        drawn = self._graph_for("A").model_dump(by_alias=True)
        drawn["edges"][0]["kind"] = "branch"
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("branch.edge_on_non_condition", _issue_codes(error))

    def test_an_empty_branches_list_is_rejected(self) -> None:
        # Normalization passes it (absent and empty must stay tellable
        # apart); this is the gate that names it — spec 1.5 / 7.4.
        drawn = self._graph_for("A").model_dump(by_alias=True)
        drawn["steps"][1]["config"]["branches"] = []
        drawn["edges"] = [drawn["edges"][0]]
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("branch.empty", _issue_codes(error))

    def test_two_default_arms_are_rejected(self) -> None:
        drawn = self._graph_for("A").model_dump(by_alias=True)
        for arm in drawn["steps"][1]["config"]["branches"]:
            arm["when"] = None
        for edge in drawn["edges"][1:]:
            edge["extensions"]["codeBridgeBranch"]["default"] = True
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("branch.duplicate_default", _issue_codes(error))

    def test_an_arm_targeting_no_step_is_rejected(self) -> None:
        drawn = self._graph_for("A").model_dump(by_alias=True)
        drawn["steps"][1]["config"]["branches"][0]["target_step_id"] = "nowhere"
        drawn["edges"][1]["toStepId"] = "nowhere"
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("branch.target_missing", _issue_codes(error))

    def test_branches_with_a_non_continue_on_success_are_rejected(self) -> None:
        drawn = self._graph_for("A").model_dump(by_alias=True)
        drawn["steps"][1]["onSuccess"] = {"type": "end"}
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("branch.on_success_conflict", _issue_codes(error))

    def test_the_linear_issue_codes_were_not_renamed(self) -> None:
        # The nine `linear.*` codes are a wire contract (routes/agents.py
        # passes them through verbatim). Widening the accepted subset must
        # not rename them, so a seq fan-out still reports the old code.
        drawn = self._graph_for("A").model_dump(by_alias=True)
        drawn["edges"].append(
            {
                "id": "run_sync:extra",
                "fromStepId": "run_sync",
                "toStepId": "notify_ok",
                "fromField": None,
                "toField": None,
                "kind": "seq",
                "extensions": {"codeBridgeLinear": {"on": "success"}},
            }
        )
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("linear.edge_unbacked", _issue_codes(error))

    def test_a_missing_arm_edge_is_reported_as_missing(self) -> None:
        drawn = self._graph_for("A").model_dump(by_alias=True)
        del drawn["edges"][2]
        error = self._rejected(Flow.model_validate(drawn))
        self.assertIn("linear.edge_missing", _issue_codes(error))


if __name__ == "__main__":
    unittest.main()
