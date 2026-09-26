"""The runner's step ordering, delegated to the kernel (agent-flow-core T-B-07).

What is being pinned
--------------------

``code_bridge_core/step_cursor.py`` stopped assuming "the next step is ``index + 1``" and
now derives the run's control graph from the rows' policies and asks
``agent_flow_core.topology.topological_sort`` to order it. The kernel is not
in the deployed server venv, so the cursor keeps the linear walk as a
permanent fallback (``_kernel_topology``'s docstring says why that is design,
not a stopgap).

Two paths answering one question is exactly the situation where a product
drifts: the kernel-backed order and the fallback order must be *the same
answer*, not two answers that usually agree. So the core of this file is an
equivalence harness — a store-free driver that walks a run purely through the
cursor and records the workflow step ids in the order they execute — run twice
over the same input, once with the kernel importable and once with it blocked
the way an uninstalled distribution blocks it. Every one of the ten stored
flow_json snapshots (``test_flow_json_snapshot_regression.SNAPSHOTS`` — the
frozen history of shapes real agents were saved in) is driven through it,
under a clean run and under a failure on each step in turn, plus forward and
backward goto flows the snapshots do not cover.

Why equality is expected rather than hoped for: in the linear+goto subset the
derived seq graph is a forward chain (a step's ``continue`` sequences into the
next row, ``goto`` edges are excluded from ordering by the kernel), and Kahn's
algorithm with the kernel's earliest-declared tie-break returns exactly the
list order for such a graph. This file is the evidence, not the argument.

The other half is the refusal path. ``topological_sort`` answers for parallel
graphs too — it linearizes them — and accepting that answer would run two
concurrent branches back to back and call it the authored workflow. So a
fan-out or a merge parks with a reason a person can act on, and that reason is
asserted here as text, not as a boolean.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.step_cursor import (  # noqa: E402
    UNSUPPORTED_TOPOLOGY_REASON,
    StepCursor,
    derive_run_graph,
    plan_execution_order,
)
from agent import task_orchestrator  # noqa: E402
from agent.task_orchestrator import _plan_workflow_steps  # noqa: E402
from code_bridge_core.workflow_v2 import normalize_workflow  # noqa: E402

# The kernel-absent simulation and the frozen stored shapes both already exist;
# re-deriving either here would let this file's idea of them drift from the
# files that own them.
from test_flow_graph_api import kernel_uninstalled  # noqa: E402
from test_flow_json_snapshot_regression import SNAPSHOTS  # noqa: E402


# ---------------------------------------------------------------------------
# A store-free driver for the ordering skeleton of
# ``task_orchestrator._drive_workflow_steps``.
#
# It runs the same loop shape — transition budget, skip walk, cursor route,
# apply route — and records only what this ticket is about: which step ran,
# in which order, and how the run ended. Everything the orchestrator does that
# is *recording* (events, step writes, parking, finishing the run) is out of
# scope here by construction; T-B-06 put that split in place and T-B-07 keeps
# it.
# ---------------------------------------------------------------------------


def _rows(flow_json: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The task step rows a run would be planned with for this flow_json."""

    rows = _plan_workflow_steps(normalize_workflow(flow_json), [])
    for index, row in enumerate(rows):
        row["id"] = f"row-{index}"
    return rows


def drive(
    steps: list[dict[str, Any]], *, failing: frozenset[str] = frozenset()
) -> list[str]:
    """Replay a run through the cursor; return its ordering trace.

    ``failing`` names the workflow step ids whose every attempt fails. The
    trace records each executed step's workflow step id in execution order,
    then a terminal marker (``end``/``abort``/``park:<reason>``/
    ``budget_exhausted``/``done``) so two paths cannot agree on the steps and
    silently disagree on how the run finished.
    """

    steps = copy.deepcopy(steps)
    cursor = StepCursor(steps)
    trace: list[str] = []

    while cursor.index < len(steps):
        if not cursor.begin_transition():
            trace.append("budget_exhausted")
            return trace

        step = steps[cursor.index]
        if cursor.should_skip(step):
            cursor.index += 1
            continue

        step_input = step["input"]
        trace.append(str(step_input.get("workflow_step_id")))

        if str(step_input.get("workflow_step_id")) in failing:
            route = cursor.route_on_failure(steps, cursor.index)
            if route.kind == "retry":
                retry_state = dict(step_input.get("retry_state") or {})
                retry_state["attempts"] = route.attempt
                step_input["retry_state"] = retry_state
                trace.append(f"retry:{route.attempt}/{route.max_attempts}")
                cursor.index = route.next_index
                continue
            if route.kind in {"continue", "goto"}:
                step["status"] = "failed"
                if route.kind == "goto":
                    trace.append(f"goto:{route.target_step_id}")
                cursor.index = route.next_index
                continue
            if route.kind == "park":
                trace.append(f"park:{route.park_reason}")
                return trace
            trace.append("abort")
            return trace

        step["status"] = "completed"
        route = cursor.advance_on_success(steps, cursor.index)
        if route.kind == "end":
            trace.append("end")
            return trace
        if route.kind == "abort":
            trace.append("abort")
            return trace
        if route.kind == "park":
            trace.append(f"park:{route.park_reason}")
            return trace
        if route.kind == "goto":
            trace.append(f"goto:{route.target_step_id}")
        cursor.index = route.next_index

    trace.append("done")
    return trace


class KernelAvailabilityTest(unittest.TestCase):
    """The two paths this file compares are really the two paths."""

    def test_the_kernel_is_installed_here(self):
        # Without this, every equivalence assertion below would be comparing
        # the fallback against itself and passing for the wrong reason.
        order = plan_execution_order(*derive_run_graph(_rows(SNAPSHOTS["rich_llm_step"])))
        self.assertEqual(order.source, "kernel")

    def test_blocking_the_kernel_reaches_the_fallback(self):
        steps = _rows(SNAPSHOTS["rich_llm_step"])
        with kernel_uninstalled():
            order = plan_execution_order(*derive_run_graph(steps))
        self.assertEqual(order.source, "fallback")
        self.assertIsNone(order.unsupported)

    def test_the_runner_keeps_routing_without_the_kernel(self):
        steps = _rows(SNAPSHOTS["goto_failure_routing"])
        with kernel_uninstalled():
            route = StepCursor(steps).advance_on_success(steps, 0)
            failure = StepCursor(steps).route_on_failure(steps, 0)
        self.assertEqual(route.kind, "advance")
        self.assertEqual(route.next_index, 1)
        self.assertEqual(failure.kind, "goto")
        self.assertEqual(failure.next_index, 2)


class SnapshotEquivalenceTest(unittest.TestCase):
    """Kernel and fallback produce the same run, over the frozen shapes."""

    def _assert_same(self, steps, *, failing=frozenset(), label=""):
        with_kernel = drive(steps, failing=failing)
        with kernel_uninstalled():
            without_kernel = drive(steps, failing=failing)
        self.assertEqual(with_kernel, without_kernel, label)
        return with_kernel

    def test_every_snapshot_runs_the_same_clean(self):
        self.assertGreaterEqual(len(SNAPSHOTS), 10)
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                steps = _rows(snapshot)
                trace = self._assert_same(steps, label=name)
                # Not a vacuous comparison: every snapshot actually runs.
                self.assertTrue([entry for entry in trace if entry not in {"done", "end"}])

    def test_every_snapshot_runs_the_same_with_each_step_failing(self):
        for name, snapshot in SNAPSHOTS.items():
            steps = _rows(snapshot)
            for row in steps:
                workflow_step_id = row["input"]["workflow_step_id"]
                with self.subTest(snapshot=name, failing=workflow_step_id):
                    self._assert_same(
                        steps,
                        failing=frozenset({workflow_step_id}),
                        label=f"{name}/{workflow_step_id}",
                    )

    def test_the_snapshot_traces_are_what_the_linear_walk_always_did(self):
        # Equality between two paths is only worth something if the shared
        # answer is still the old one. `legacy_shell_with_offschema_fields`
        # is the escalation shape: success ends the run, failure jumps to the
        # diagnosis step that must not run on a clean night.
        steps = _rows(SNAPSHOTS["legacy_shell_with_offschema_fields"])
        self.assertEqual(self._assert_same(steps), ["run_cycle", "end"])
        self.assertEqual(
            self._assert_same(steps, failing=frozenset({"run_cycle"})),
            ["run_cycle", "goto:diagnose", "diagnose", "done"],
        )


class GotoEquivalenceTest(unittest.TestCase):
    """Forward and backward jumps agree — including the one no snapshot has."""

    FORWARD = [
        {
            "id": "check",
            "type": "llm",
            "name": "Check",
            "on_success": {"type": "goto_step", "target_step_id": "report"},
        },
        {"id": "repair", "type": "shell", "name": "Repair", "script_id": "s1"},
        {"id": "report", "type": "notify", "name": "Report"},
    ]

    #: A backward jump is the case the kernel's goto exclusion exists for: as
    #: an ordering dependency `retry -> check` would close a cycle and
    #: `topological_sort` would raise, taking the run down. As a goto edge it
    #: is a jump the runner takes at run time.
    BACKWARD = [
        {
            "id": "check",
            "type": "llm",
            "name": "Check",
            "on_failure": {"type": "goto_step", "target_step_id": "repair"},
        },
        {
            "id": "repair",
            "type": "shell",
            "name": "Repair",
            "script_id": "s1",
            "on_success": {"type": "goto_step", "target_step_id": "check"},
        },
        {"id": "report", "type": "notify", "name": "Report"},
    ]

    def _assert_same(self, steps, *, failing=frozenset()):
        with_kernel = drive(steps, failing=failing)
        with kernel_uninstalled():
            without_kernel = drive(steps, failing=failing)
        self.assertEqual(with_kernel, without_kernel)
        return with_kernel

    def test_a_forward_success_jump_skips_the_middle_step(self):
        self.assertEqual(
            self._assert_same(_rows(self.FORWARD)),
            ["check", "goto:report", "report", "done"],
        )

    def test_a_backward_jump_is_not_a_cycle(self):
        steps = _rows(self.BACKWARD)
        order = plan_execution_order(*derive_run_graph(steps))
        self.assertEqual(order.source, "kernel")
        self.assertIsNone(
            order.unsupported,
            "a backward goto is a control jump, not an ordering dependency",
        )
        self.assertEqual(order.positions, (0, 1, 2))

    def test_a_backward_jump_runs_the_same_both_ways(self):
        # check fails -> repair -> jumps back to check, which is failed rather
        # than completed and so runs again -> fails again -> jumps to repair,
        # which *is* completed and is walked past -> report. The loop settles
        # because a completed step is skipped, and both paths settle the same.
        self.assertEqual(
            self._assert_same(_rows(self.BACKWARD), failing=frozenset({"check"})),
            [
                "check",
                "goto:repair",
                "repair",
                "goto:check",
                "check",
                "goto:repair",
                "report",
                "done",
            ],
        )

    def test_a_backward_jump_that_settles_runs_the_same_both_ways(self):
        self.assertEqual(
            self._assert_same(_rows(self.BACKWARD)),
            # check succeeds; repair jumps back to check, which is already
            # completed and therefore skipped; report runs.
            ["check", "repair", "goto:check", "report", "done"],
        )


class UnsupportedTopologyTest(unittest.TestCase):
    """A shape this runner cannot execute stops loudly, with instructions."""

    def _fan_out_rows(self):
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
        return rows

    def _merge_rows(self):
        rows = _rows(
            [
                {"id": "left", "type": "llm", "name": "Left"},
                {"id": "right", "type": "llm", "name": "Right"},
                {"id": "join", "type": "llm", "name": "Join"},
            ]
        )
        rows[0]["input"]["on_success"] = {"type": "parallel", "targets": ["join"]}
        return rows

    def test_a_parallel_branch_is_refused_not_linearized(self):
        rows = self._fan_out_rows()
        order = plan_execution_order(*derive_run_graph(rows))
        self.assertIsNotNone(order.unsupported)
        reason = order.unsupported
        # The reason has to name the offending step, say why this runner
        # cannot run it, and say what to change — a person reads this in the
        # checkpoint prompt with no other context.
        self.assertIn("'fan'", reason)
        self.assertIn("2 steps at once", reason)
        self.assertIn("'left'", reason)
        self.assertIn("'right'", reason)
        self.assertIn("one step at a time", reason)
        self.assertIn("Edit the workflow", reason)

    def test_a_merge_is_refused(self):
        order = plan_execution_order(*derive_run_graph(self._merge_rows()))
        self.assertIsNotNone(order.unsupported)
        self.assertIn("join back into", order.unsupported)
        self.assertIn("'join'", order.unsupported)
        self.assertIn("Edit the workflow", order.unsupported)

    def test_the_cursor_parks_on_success_instead_of_walking_the_list(self):
        rows = self._fan_out_rows()
        route = StepCursor(rows).advance_on_success(rows, 0)
        self.assertEqual(route.kind, "park")
        self.assertEqual(route.park_reason, UNSUPPORTED_TOPOLOGY_REASON)
        self.assertIn("one step at a time", route.park_prompt or "")
        self.assertIsNone(
            route.next_index, "a parked route must not also hand out a next step"
        )

    def test_the_cursor_parks_on_failure_too(self):
        rows = self._fan_out_rows()
        route = StepCursor(rows).route_on_failure(rows, 0)
        self.assertEqual(route.kind, "park")
        self.assertEqual(route.park_reason, UNSUPPORTED_TOPOLOGY_REASON)
        self.assertIn("Edit the workflow", route.park_prompt or "")

    def test_the_run_parks_rather_than_running_the_branches_in_a_row(self):
        trace = drive(self._fan_out_rows())
        self.assertEqual(trace, ["fan", f"park:{UNSUPPORTED_TOPOLOGY_REASON}"])

    def test_the_refusal_does_not_depend_on_the_kernel(self):
        rows = self._fan_out_rows()
        with kernel_uninstalled():
            trace = drive(rows)
            order = plan_execution_order(*derive_run_graph(rows))
        self.assertEqual(trace, ["fan", f"park:{UNSUPPORTED_TOPOLOGY_REASON}"])
        self.assertIsNotNone(order.unsupported)

    def test_an_ordering_cycle_names_the_steps_and_the_fix(self):
        # Reached through the same list-valued successor shape: `a` and `b`
        # sequence into each other, which is a dependency loop rather than a
        # goto, so there is no order to run them in.
        rows = _rows(
            [
                {"id": "a", "type": "llm", "name": "A"},
                {"id": "b", "type": "llm", "name": "B"},
            ]
        )
        rows[0]["input"]["on_success"] = {"type": "parallel", "targets": ["b"]}
        rows[1]["input"]["on_success"] = {"type": "parallel", "targets": ["a"]}
        order = plan_execution_order(*derive_run_graph(rows))
        self.assertEqual(order.source, "kernel")
        self.assertIsNotNone(order.unsupported)
        self.assertIn("loop", order.unsupported)
        self.assertIn("goto_step", order.unsupported)


class OrchestratorParkWiringTest(unittest.TestCase):
    """The cursor decides; the orchestrator still does the recording.

    T-B-06 split "where execution goes" from "what gets written down", and
    delegating the first half must not move the second. So the park route is
    checked here at the seam: the orchestrator turns it into the same park it
    performs for ask_user / manual_handoff, and does not finish the run.
    """

    def test_a_topology_park_is_recorded_as_a_park(self):
        rows = UnsupportedTopologyTest()._fan_out_rows()
        with mock.patch.object(
            task_orchestrator, "get_agent_store", return_value=mock.MagicMock()
        ), mock.patch.object(
            task_orchestrator, "_wait_for_user_step"
        ) as park, mock.patch.object(
            task_orchestrator, "_finish_workflow_execution"
        ) as finish:
            next_index = task_orchestrator._apply_workflow_success_policy(
                task_id="task_1",
                run_id="run_1",
                steps=rows,
                completed_index=0,
            )

        self.assertIsNone(next_index, "the run must not walk on to the next row")
        finish.assert_not_called()
        park.assert_called_once()
        kwargs = park.call_args.kwargs
        self.assertEqual(kwargs["reason"], UNSUPPORTED_TOPOLOGY_REASON)
        self.assertEqual(kwargs["step"]["id"], rows[0]["id"])
        self.assertIn("one step at a time", kwargs["prompt"])

    def test_the_checkpoint_tells_the_user_to_fix_the_workflow(self):
        # Parking with the generic "complete the manual action" instruction
        # would send a person looking for a task that does not exist.
        action = task_orchestrator._required_user_action(UNSUPPORTED_TOPOLOGY_REASON)
        self.assertIn("워크플로", action)
        self.assertNotEqual(
            action, task_orchestrator._required_user_action("manual_handoff")
        )


class DerivationTest(unittest.TestCase):
    """The graph handed to the kernel is the one the policies describe."""

    def test_goto_edges_are_marked_so_the_kernel_excludes_them(self):
        steps = _rows(SNAPSHOTS["goto_failure_routing"])
        _, edges = derive_run_graph(steps)
        goto = [edge for edge in edges if edge["kind"] == "goto"]
        self.assertEqual(len(goto), 1)
        self.assertEqual(goto[0], {
            "fromStepId": "0",
            "toStepId": "2",
            "kind": "goto",
            "on": "failure",
        })

    def test_an_end_policy_draws_no_sequential_edge(self):
        steps = _rows(SNAPSHOTS["legacy_shell_with_offschema_fields"])
        _, edges = derive_run_graph(steps)
        self.assertEqual(
            [edge for edge in edges if edge["kind"] == "seq"],
            [],
            "a step that ends the workflow does not sequence into the next one",
        )

    def test_a_retry_chain_ending_in_a_goto_is_a_goto_edge(self):
        rows = _rows(
            [
                {
                    "id": "flaky",
                    "type": "llm",
                    "name": "Flaky",
                    "on_failure": {
                        "type": "retry",
                        "max_attempts": 1,
                        "then": {"type": "goto_step", "target_step_id": "rescue"},
                    },
                },
                {"id": "rescue", "type": "llm", "name": "Rescue"},
            ]
        )
        _, edges = derive_run_graph(rows)
        self.assertIn(
            {"fromStepId": "0", "toStepId": "1", "kind": "goto", "on": "failure"},
            edges,
        )

    def test_failure_continue_after_a_goto_success_draws_the_walk(self):
        # E6 (LINEAR_FLOW_MAPPING 3.0.1): the morning-check shape. Success
        # jumps over the diagnosis rows; failure walks into them. Before, no
        # edge described that walk, so row 2 had nothing leading to it.
        rows = _rows(
            [
                {"id": "git", "type": "llm", "name": "Git"},
                {
                    "id": "tests",
                    "type": "llm",
                    "name": "Tests",
                    "on_success": {"type": "goto_step", "target_step_id": "pass"},
                    "on_failure": {"type": "continue"},
                },
                {"id": "analyze", "type": "llm", "name": "Analyze",
                 "on_success": {"type": "end"}},
                {"id": "pass", "type": "llm", "name": "Pass",
                 "on_success": {"type": "end"}},
            ]
        )
        nodes, edges = derive_run_graph(rows)
        self.assertEqual(
            [edge for edge in edges if edge["fromStepId"] == "1"],
            [
                {"fromStepId": "1", "toStepId": "3", "kind": "goto", "on": "success"},
                {"fromStepId": "1", "toStepId": "2", "kind": "seq", "on": "failure"},
            ],
        )
        # Still the runner's order, and still executable one row at a time.
        order = plan_execution_order(nodes, edges)
        self.assertIsNone(order.unsupported)
        self.assertEqual(order.positions, (0, 1, 2, 3))

    def test_failure_continue_beside_a_seq_success_draws_nothing_extra(self):
        rows = _rows(
            [
                {"id": "a", "type": "llm", "name": "A",
                 "on_failure": {"type": "continue"}},
                {"id": "b", "type": "llm", "name": "B"},
            ]
        )
        _, edges = derive_run_graph(rows)
        self.assertEqual(
            [edge for edge in edges if edge["fromStepId"] == "0"],
            [{"fromStepId": "0", "toStepId": "1", "kind": "seq", "on": "success"}],
        )

    def test_rows_the_loop_skips_still_sequence(self):
        # A non-workflow row (no workflow_step_id) is walked past, and the
        # walk is what the edge describes — otherwise the graph would claim
        # the run stops at it.
        rows = _rows(SNAPSHOTS["app_action_device_flow"])
        rows.insert(1, {"id": "row-x", "status": "queued", "input": {}})
        nodes, edges = derive_run_graph(rows)
        self.assertEqual(len(nodes), 3)
        self.assertIn(
            {"fromStepId": "1", "toStepId": "2", "kind": "seq", "on": "success"},
            edges,
        )
        order = plan_execution_order(nodes, edges)
        self.assertEqual(order.positions, (0, 1, 2))
        self.assertIsNone(order.unsupported)

    def test_an_unroutable_policy_type_still_walks_the_list(self):
        # No behavior change where there is no topology claim: a policy type
        # this runner has no route for named no successors, so it keeps the
        # step-to-next-step walk it has always had.
        rows = _rows(SNAPSHOTS["app_action_device_flow"])
        rows[0]["input"]["on_success"] = {"type": "teleport"}
        route = StepCursor(rows).advance_on_success(rows, 0)
        self.assertEqual(route.kind, "advance")
        self.assertEqual(route.next_index, 1)


if __name__ == "__main__":
    unittest.main()
