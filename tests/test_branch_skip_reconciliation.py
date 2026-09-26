"""The arm that was not taken is settled, and says why (T-H-11).

Step rows are all created ``queued`` before a run starts, so a branch leaves
the rows of the arm it did not take queued for ever. The run finishing is
safe — only ``failed`` rows are counted — but the record lies: the app draws
``queued`` as "about to run", so a finished run displays steps that will never
happen and cannot answer "why didn't the restock step happen?".

Two properties carry this file, and they pull in opposite directions:

* every row a branch put out of reach is settled as ``skipped``, carrying the
  condition and the arm that was taken instead;
* **the branch attribution claims nothing else.** A linear workflow walks
  every row, so it can never produce a ``skipped`` one
  (RUNNER_BRANCHING_SPEC 8) — that is the regression net for every workflow
  that exists today.

A run that stopped early for some *other* reason — an abort, an
``on_success: end`` above the row — is settled too, but by a different
function and under a different reason (``run_ended`` rather than
``branch_not_taken``), because only one of the two can name a condition. The
tests at the bottom of this file hold that separation: the rows are settled,
and the record does not claim a branch decided them.
"""

from __future__ import annotations

import asyncio
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store  # noqa: E402
from agent import script_store as script_store_module  # noqa: E402
from code_bridge_core.step_cursor import unreachable_by_branch  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402


def _row(
    workflow_step_id: str,
    *,
    status: str = "queued",
    workflow_type: str = "notify",
    branches: list | None = None,
    on_success: dict | None = None,
    condition: dict | None = None,
):
    step_input: dict = {
        "workflow_step_id": workflow_step_id,
        "workflow_type": workflow_type,
    }
    if branches is not None:
        step_input["branches"] = branches
    if on_success is not None:
        step_input["on_success"] = on_success
    row: dict = {
        "id": f"row_{workflow_step_id}",
        "run_id": "run_1",
        "title": workflow_step_id,
        "status": status,
        "input": step_input,
    }
    if condition is not None:
        row["output"] = {"condition": condition}
    return row


def _arm(label: str, target: str):
    return {"label": label, "when": None, "target_step_id": target}


def _took(index: int, label: str, target: str, default: bool = False):
    return {
        "matched_index": index,
        "matched_label": label,
        "target_step_id": target,
        "default": default,
        "op": None,
        "resolved": None,
        "evaluated": [],
    }


class WhichRowsABranchPutOutOfReachTest(unittest.TestCase):
    """The pure half. No store, no run — just the two reachabilities."""

    def test_a_list_with_no_condition_loses_nothing(self):
        steps = [_row("a", status="completed"), _row("b", status="completed")]
        self.assertEqual(unreachable_by_branch(steps), {})

    def test_a_condition_that_has_not_decided_yet_loses_nothing(self):
        steps = [
            _row(
                "check",
                workflow_type="condition",
                branches=[_arm("x", "left"), _arm("y", "right")],
            ),
            _row("left"),
            _row("right"),
        ]
        self.assertEqual(unreachable_by_branch(steps), {})

    def test_a_condition_with_no_branches_loses_nothing(self):
        # Every workflow saved before track H. It writes a string output, not a
        # decision, and nothing downstream of it is ever called skipped.
        steps = [
            _row("check", workflow_type="condition", status="completed"),
            _row("next", status="completed"),
        ]
        self.assertEqual(unreachable_by_branch(steps), {})

    def test_the_un_taken_arms_target_is_named_with_the_arm_taken_instead(self):
        steps = [
            _row(
                "check",
                status="completed",
                workflow_type="condition",
                branches=[_arm("실패", "diagnose"), _arm("성공", "notify_ok")],
                condition=_took(0, "실패", "diagnose"),
            ),
            _row("diagnose", status="completed", on_success={"type": "end"}),
            _row("notify_ok", on_success={"type": "end"}),
        ]
        lost = unreachable_by_branch(steps)
        self.assertEqual(list(lost), [2])
        self.assertEqual(lost[2]["condition_workflow_step_id"], "check")
        self.assertEqual(lost[2]["matched_label"], "실패")
        self.assertEqual(lost[2]["target_step_id"], "diagnose")
        self.assertFalse(lost[2]["default"])

    def test_rows_only_reachable_through_the_un_taken_arm_go_too(self):
        # The whole arm, not just the row the arm points at — otherwise the
        # second step of a two-step arm stays queued and the record still
        # cannot be read straight through.
        steps = [
            _row(
                "check",
                status="completed",
                workflow_type="condition",
                branches=[_arm("A", "restock"), _arm("B", "notify_ok")],
                condition=_took(1, "B", "notify_ok"),
            ),
            _row("restock"),
            _row("tell_buyer", on_success={"type": "end"}),
            _row("notify_ok", status="completed", on_success={"type": "end"}),
        ]
        lost = unreachable_by_branch(steps)
        self.assertEqual(sorted(lost), [1, 2])
        self.assertEqual(
            {position: lost[position]["target_step_id"] for position in lost},
            {1: "notify_ok", 2: "notify_ok"},
        )

    def test_a_row_that_was_out_of_reach_anyway_is_not_blamed_on_the_branch(self):
        # `orphan` sits after a step whose `on_success` is `end`, so no
        # decision could ever have reached it. It was queued at the end of a
        # run before branching existed and it stays that way.
        steps = [
            _row(
                "check",
                status="completed",
                workflow_type="condition",
                branches=[_arm("A", "left"), _arm("B", "right")],
                condition=_took(0, "A", "left"),
            ),
            _row("left", status="completed", on_success={"type": "end"}),
            _row("right", on_success={"type": "end"}),
            _row("orphan"),
        ]
        self.assertEqual(sorted(unreachable_by_branch(steps)), [2])

    def test_the_earliest_condition_that_could_have_reached_the_row_owns_it(self):
        steps = [
            _row(
                "first",
                status="completed",
                workflow_type="condition",
                branches=[_arm("A", "shared"), _arm("B", "second")],
                condition=_took(1, "B", "second"),
            ),
            _row(
                "second",
                status="completed",
                workflow_type="condition",
                branches=[_arm("C", "shared"), _arm("D", "tail")],
                condition=_took(1, "D", "tail"),
            ),
            _row("shared", on_success={"type": "end"}),
            _row("tail", status="completed", on_success={"type": "end"}),
        ]
        lost = unreachable_by_branch(steps)
        self.assertEqual(sorted(lost), [2])
        self.assertEqual(lost[2]["condition_workflow_step_id"], "first")


class WhatARunLeavesBehindTest(unittest.TestCase):
    """The end-to-end half, on the path a schedule takes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "skip.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        script_store_module._script_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = agent_store.get_agent_store()

        script_path = self.dir / "sync.sh"
        script_path.write_text('#!/bin/bash\necho "SYNC OUTPUT"\nexit "${1:-0}"\n')
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
        self.script = script_store_module.get_script_store().register(
            name="sync_devices", path=str(script_path)
        )

    def _restore(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        script_store_module._script_store = None
        database.DB_PATH = self._original_db_path

    def _drive(self, flow_json):
        agent = self.store.create_agent(
            name="skip bot",
            system_prompt="Run workflow steps.",
            provider_id="openai",
            flow_json=flow_json,
        )
        task = self.store.create_task(
            title="Run workflow", assigned_agent_id=agent["id"], goal="g"
        )
        result = prepare_task_orchestration(
            task["id"], provider_id="openai", auto_start=False
        )
        assert result is not None
        asyncio.run(execute_task_orchestration(result["execution"]))
        rows = {
            (row.get("input") or {}).get("workflow_step_id"): row
            for row in self.store.list_task_steps(task["id"])
        }
        events = [
            event.get("event_type")
            for event in self.store.list_events(result["run"]["id"])
        ]
        return rows, events

    def _shell(self, step_id, exit_code, **extra):
        step = {
            "id": step_id,
            "type": "shell",
            "name": step_id,
            "script_id": self.script["id"],
            "script_args": [exit_code],
        }
        step.update(extra)
        return step

    @staticmethod
    def _notify(step_id, **extra):
        step = {
            "id": step_id,
            "type": "notify",
            "name": step_id,
            "notify": {"title": step_id, "body": step_id},
        }
        step.update(extra)
        return step

    def _branching_flow(self, exit_code, *, default_arm=False):
        second = (
            {"label": "그 외", "when": None, "target_step_id": "notify_ok"}
            if default_arm
            else {
                "label": "성공",
                "when": {
                    "left": "{{steps.run_sync.exit_code}}",
                    "op": "equals",
                    "right": "0",
                },
                "target_step_id": "notify_ok",
            }
        )
        return [
            self._shell(
                "run_sync",
                exit_code,
                on_failure={"type": "continue"},
                on_success={"type": "continue"},
            ),
            {
                "id": "check_exit",
                "type": "condition",
                "name": "종료 코드 확인",
                "branches": [
                    {
                        "label": "실패",
                        "when": {
                            "left": "{{steps.run_sync.exit_code}}",
                            "op": "not_equals",
                            "right": "0",
                        },
                        "target_step_id": "diagnose",
                    },
                    second,
                ],
            },
            self._notify("diagnose", on_success={"type": "end"}),
            self._notify("notify_ok", on_success={"type": "end"}),
        ]

    # -- the arm that was not taken ----------------------------------------

    def test_the_un_taken_arm_is_settled_and_the_row_says_why(self):
        rows, events = self._drive(self._branching_flow("1"))
        self.assertEqual(rows["diagnose"]["status"], "completed")
        self.assertEqual(rows["notify_ok"]["status"], "skipped")
        record = (rows["notify_ok"]["output"] or {})["skipped"]
        self.assertEqual(record["reason"], "branch_not_taken")
        self.assertEqual(record["condition_step_id"], "check_exit")
        self.assertEqual(record["matched_index"], 0)
        self.assertEqual(record["matched_label"], "실패")
        self.assertEqual(record["target_step_id"], "diagnose")
        self.assertFalse(record["default"])
        # Readable without reconstructing the run: which condition, which arm,
        # where it went instead.
        self.assertIn("check_exit", record["message"])
        self.assertIn("실패", record["message"])
        self.assertIn("diagnose", record["message"])
        self.assertIn("task.step.skipped", events)

    def test_a_settled_row_is_terminal_and_cannot_be_mistaken_for_pending(self):
        rows, _events = self._drive(self._branching_flow("1"))
        self.assertIsNotNone(
            rows["notify_ok"]["ended_at"], "a skipped row is finished with the run"
        )
        self.assertEqual(
            [
                workflow_step_id
                for workflow_step_id, row in rows.items()
                if row["status"] == "queued"
            ],
            [],
            "a finished branching run leaves nothing claiming to be pending",
        )

    def test_the_other_arm_settles_the_row_the_branch_jumped_over(self):
        rows, _events = self._drive(self._branching_flow("0"))
        self.assertEqual(rows["notify_ok"]["status"], "completed")
        self.assertEqual(rows["diagnose"]["status"], "skipped")
        self.assertEqual(
            (rows["diagnose"]["output"] or {})["skipped"]["target_step_id"],
            "notify_ok",
        )

    def test_the_default_arm_says_it_was_the_default(self):
        rows, _events = self._drive(self._branching_flow("0", default_arm=True))
        record = (rows["diagnose"]["output"] or {})["skipped"]
        self.assertTrue(record["default"])
        self.assertEqual(record["matched_label"], "그 외")
        self.assertIn("기본 가지", record["message"])

    # -- and nothing else --------------------------------------------------

    def test_a_linear_workflow_produces_no_skipped_row(self):
        # RUNNER_BRANCHING_SPEC 8's invariant, and the regression net for every
        # workflow that exists today: a linear run walks every row.
        rows, events = self._drive(
            [
                self._shell("run_sync", "0", on_success={"type": "continue"}),
                self._notify("notify_ok"),
            ]
        )
        self.assertEqual(
            [row["status"] for row in rows.values()], ["completed", "completed"]
        )
        self.assertNotIn("task.step.skipped", events)

    def test_a_condition_without_branches_produces_no_skipped_row(self):
        rows, events = self._drive(
            [
                self._shell("run_sync", "0", on_success={"type": "continue"}),
                {"id": "check_exit", "type": "condition", "name": "확인"},
                self._notify("notify_ok"),
            ]
        )
        self.assertEqual(
            (rows["check_exit"]["output"] or {}).get("result"),
            "condition step completed without branching",
        )
        self.assertNotIn("task.step.skipped", events)
        self.assertNotIn("skipped", [row["status"] for row in rows.values()])

    def test_an_aborted_runs_unreached_rows_say_the_run_ended(self):
        # These rows were reachable on the arm the run *did* take; it stopped
        # short of them. Nothing about a branch put them out of reach — so the
        # branch attribution must not claim them, and the assertion below on
        # `reason` is what holds that line.
        #
        # This test used to assert they stayed `queued`, on the grounds that
        # settling them would change what an aborting workflow reports. It
        # does change it. `queued` on a run that is over is not a neutral
        # placeholder: the app draws it as *about to run*, so the old record
        # was not silent about these rows, it was wrong about them. What must
        # stay true — and does, one assertion down — is that the reason is
        # `run_ended` and names no condition.
        rows, events = self._drive(
            [
                self._shell(
                    "run_sync",
                    "1",
                    on_failure={"type": "continue"},
                    on_success={"type": "continue"},
                ),
                {
                    "id": "check_exit",
                    "type": "condition",
                    "name": "확인",
                    "branches": [
                        {
                            "label": "실패",
                            "when": {
                                "left": "{{steps.run_sync.exit_code}}",
                                "op": "not_equals",
                                "right": "0",
                            },
                            "target_step_id": "diagnose",
                        },
                        {"label": "그 외", "when": None, "target_step_id": "notify_ok"},
                    ],
                },
                self._shell(
                    "diagnose",
                    "1",
                    on_failure={"type": "abort"},
                    on_success={"type": "continue"},
                ),
                self._notify("after_diagnose"),
                self._notify("notify_ok", on_success={"type": "end"}),
            ]
        )
        self.assertEqual(rows["diagnose"]["status"], "failed")
        self.assertEqual(rows["after_diagnose"]["status"], "skipped")
        self.assertEqual(rows["notify_ok"]["status"], "skipped")

        for step_id in ("after_diagnose", "notify_ok"):
            record = (rows[step_id]["output"] or {})["skipped"]
            self.assertEqual(record["reason"], "run_ended", step_id)
            self.assertEqual(record["run_status"], "failed", step_id)
            # Where the run stopped, named — the sentence has to be readable
            # off this row alone, exactly as the branch one is.
            self.assertIn("diagnose", record["message"], step_id)
            # And no condition is blamed: these rows were never on an un-taken
            # arm, so every branch field the app reads must be absent.
            for branch_field in (
                "condition_step_id",
                "matched_label",
                "target_step_id",
                "default",
            ):
                self.assertNotIn(branch_field, record, f"{step_id}.{branch_field}")

        self.assertIn("task.step.skipped", events)

    def test_a_completed_run_that_ended_early_settles_the_rows_below_it(self):
        """The shape the user's own agent produced, and the complaint about it.

        A linear flow whose first step ends the run on success: the two steps
        under it were built for the failure path, the run went nowhere near
        them, and they sat on ``queued`` — which the app draws as *about to
        run* — on a run that was over. No condition is involved anywhere in
        this flow, so the branch attribution has nothing to say about it and
        the rows stayed stranded for as long as the record existed.
        """
        rows, events = self._drive(
            [
                self._shell("check_disk", "0", on_success={"type": "end"}),
                self._notify("diagnose_failure"),
                self._notify("notify_result"),
            ]
        )
        self.assertEqual(rows["check_disk"]["status"], "completed")
        self.assertEqual(rows["diagnose_failure"]["status"], "skipped")
        self.assertEqual(rows["notify_result"]["status"], "skipped")

        record = (rows["diagnose_failure"]["output"] or {})["skipped"]
        self.assertEqual(record["reason"], "run_ended")
        self.assertEqual(record["run_status"], "completed")
        self.assertEqual(record["last_workflow_step_id"], "check_disk")
        self.assertIn("check_disk", record["message"])
        self.assertIn("task.step.skipped", events)

    def test_a_row_that_actually_ran_is_never_overwritten(self):
        """Only the absence of a run is settled, never the record of one.

        The distinction this holds is the one that makes the correction safe:
        a `completed` row on a run that ended early keeps its own status and
        its own output. If this ever fails, the settlement is writing over
        evidence rather than filling in a blank.
        """
        rows, _events = self._drive(
            [
                self._shell("run_sync", "0", on_success={"type": "continue"}),
                self._notify("announce", on_success={"type": "end"}),
                self._notify("never_reached"),
            ]
        )
        self.assertEqual(rows["run_sync"]["status"], "completed")
        self.assertEqual(rows["announce"]["status"], "completed")
        self.assertNotIn("skipped", rows["run_sync"]["output"] or {})
        self.assertEqual(rows["never_reached"]["status"], "skipped")
        # The run stopped at `announce`, not at the last step that merely
        # exists in the definition.
        record = (rows["never_reached"]["output"] or {})["skipped"]
        self.assertEqual(record["last_workflow_step_id"], "announce")


if __name__ == "__main__":
    unittest.main()
