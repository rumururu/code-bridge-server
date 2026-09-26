"""A condition step branches — on **both** dispatch paths, end to end.

`condition` is dispatched in two places: `_execute_single_workflow_task_step`
(the "run this one step" button) and `_drive_workflow_steps` (the auto-advance
loop every schedule uses). This project has already paid for that shape once:
`mcp_tool` was wired into one site while the other kept a stale reference, and
1800 passing tests did not catch it, because not one of them drove the second
path. `shell` shipped the same way — working by hand, stalling on a schedule.

So the test that matters here is not "do both call the same function" (that is
a property a review checks, and a review is what missed `mcp_tool`). It is
this: **run the same branching workflow down each path and compare where it
lands.** Wire only one site and this file goes red.

RUNNER_BRANCHING_SPEC 9.1, T-H-07.
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
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    execute_task_step_adapter,
    prepare_task_orchestration,
)
from core import database  # noqa: E402


def _exit_code_flow(script_id: str, exit_code: str) -> list[dict]:
    """Spec example A: a script's exit code chooses one of two arms.

    The primary use case of the whole feature — `on_failure: continue` on a
    shell step, then a condition that reads the exit code the failed row
    still carries.
    """

    return [
        {
            "id": "run_sync",
            "type": "shell",
            "name": "동기화 스크립트",
            "script_id": script_id,
            "script_args": [exit_code],
            "on_failure": {"type": "continue"},
            "on_success": {"type": "continue"},
        },
        {
            "id": "check_exit",
            "type": "condition",
            "name": "종료 코드 확인",
            "description": "스크립트 종료 코드로 갈 길을 정한다.",
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
                {
                    "label": "성공",
                    "when": {
                        "left": "{{steps.run_sync.exit_code}}",
                        "op": "equals",
                        "right": "0",
                    },
                    "target_step_id": "notify_ok",
                },
            ],
        },
        {
            "id": "diagnose",
            "type": "notify",
            "name": "실패 알림",
            "notify": {"title": "동기화 실패", "body": "스크립트가 실패했습니다."},
            "on_success": {"type": "end"},
        },
        {
            "id": "notify_ok",
            "type": "notify",
            "name": "성공 알림",
            "notify": {"title": "동기화 완료", "body": "정상 종료했습니다."},
            "on_success": {"type": "end"},
        },
    ]


class ConditionRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "condition_runner.db"
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

        # A poll that eventually settles: prints RUNNING twice, then DONE.
        # A backward arm can only be shown to *loop* against a predicate that
        # stops holding — one that never does only shows the budget working.
        self.counter = self.dir / "poll_count"
        poll_path = self.dir / "poll.sh"
        poll_path.write_text(
            "#!/bin/bash\n"
            'n=$(cat "$1" 2>/dev/null || echo 0)\n'
            "n=$((n+1))\n"
            'echo "$n" > "$1"\n'
            'if [ "$n" -ge 3 ]; then echo "DONE"; else echo "RUNNING"; fi\n'
            "exit 0\n"
        )
        poll_path.chmod(poll_path.stat().st_mode | stat.S_IEXEC)
        self.poll_script = script_store_module.get_script_store().register(
            name="poll_job", path=str(poll_path)
        )

    def _restore(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        script_store_module._script_store = None
        database.DB_PATH = self._original_db_path

    def _prepare(self, flow_json):
        agent = self.store.create_agent(
            name="branch bot",
            system_prompt="Run workflow steps.",
            provider_id="openai",
            flow_json=flow_json,
        )
        task = self.store.create_task(
            title="Run branching workflow",
            assigned_agent_id=agent["id"],
            goal="Take the right arm.",
        )
        result = prepare_task_orchestration(
            task["id"], provider_id="openai", auto_start=False
        )
        assert result is not None
        return task, result

    def _rows(self, task_id):
        return {
            (row.get("input") or {}).get("workflow_step_id"): row
            for row in self.store.list_task_steps(task_id)
        }

    # -- the two dispatch paths --------------------------------------------

    def _drive_auto_advance(self, flow_json):
        """The path every schedule takes."""
        task, result = self._prepare(flow_json)
        asyncio.run(execute_task_orchestration(result["execution"]))
        return task, self._rows(task["id"])

    def _drive_single_steps(self, flow_json):
        """The path the 'run this step' button takes, step by step by hand."""
        task, _result = self._prepare(flow_json)
        for workflow_step_id in ("run_sync", "check_exit"):
            row = self._rows(task["id"])[workflow_step_id]
            asyncio.run(execute_task_step_adapter(task["id"], row["id"]))
        return task, self._rows(task["id"])

    def _condition_record(self, rows):
        return ((rows["check_exit"].get("output") or {}).get("condition")) or {}

    def _terminating_polling_flow(self):
        """Spec example C: poll until the job is done, then carry on."""

        return [
            {
                "id": "poll",
                "type": "shell",
                "name": "작업 상태 조회",
                "script_id": self.poll_script["id"],
                "script_args": [str(self.counter)],
                "on_success": {"type": "continue"},
            },
            {
                "id": "still_running",
                "type": "condition",
                "name": "아직 실행 중인가",
                "branches": [
                    {
                        "label": "아직 실행 중",
                        "when": {
                            "left": "{{steps.poll.stdout}}",
                            "op": "contains",
                            "right": "RUNNING",
                        },
                        "target_step_id": "poll",
                    },
                    {"label": "끝났다", "when": None, "target_step_id": "report"},
                ],
            },
            {
                "id": "report",
                "type": "notify",
                "name": "결과 보고",
                "notify": {"title": "끝", "body": "끝"},
                "on_success": {"type": "end"},
            },
        ]


class BothPathsBranchTest(ConditionRunnerTestCase):
    """The regression net for RUNNER_BRANCHING_SPEC 9.1.

    Each test drives *both* dispatch sites and compares. Wiring only one is a
    failure here even though the workflow, the evaluator and the cursor are
    all correct.
    """

    def _both(self, exit_code):
        flow = _exit_code_flow(self.script["id"], exit_code)
        _task_a, auto = self._drive_auto_advance(flow)
        _task_b, single = self._drive_single_steps(flow)
        return auto, single

    def test_the_failure_arm_is_taken_by_both_paths(self):
        auto, single = self._both("1")
        self.assertEqual(
            self._condition_record(auto).get("target_step_id"),
            "diagnose",
            "the auto-advance loop did not evaluate the condition — this is the"
            " mcp_tool failure repeating",
        )
        self.assertEqual(
            self._condition_record(single).get("target_step_id"), "diagnose"
        )
        self.assertEqual(
            self._condition_record(auto).get("target_step_id"),
            self._condition_record(single).get("target_step_id"),
        )

    def test_the_success_arm_is_taken_by_both_paths(self):
        auto, single = self._both("0")
        self.assertEqual(
            self._condition_record(auto).get("target_step_id"), "notify_ok"
        )
        self.assertEqual(
            self._condition_record(single).get("target_step_id"), "notify_ok"
        )

    def test_the_two_paths_record_identical_reasoning(self):
        auto, single = self._both("1")
        self.assertEqual(self._condition_record(auto), self._condition_record(single))


class TheRunActuallyGoesThereTest(ConditionRunnerTestCase):
    """Recording an arm is not enough — the run has to walk down it."""

    def test_a_nonzero_exit_runs_diagnose_and_not_notify_ok(self):
        _task, rows = self._drive_auto_advance(
            _exit_code_flow(self.script["id"], "1")
        )
        self.assertEqual(rows["run_sync"]["status"], "failed")
        self.assertEqual(rows["check_exit"]["status"], "completed")
        self.assertEqual(rows["diagnose"]["status"], "completed")
        # Was `queued` until T-H-11. The arm was not taken, and a run that has
        # finished must not display a row as though it were about to run
        # (RUNNER_BRANCHING_SPEC 8).
        self.assertEqual(
            rows["notify_ok"]["status"],
            "skipped",
            "the arm that was not taken must not have run, and must say so",
        )

    def test_a_zero_exit_runs_notify_ok_and_not_diagnose(self):
        _task, rows = self._drive_auto_advance(
            _exit_code_flow(self.script["id"], "0")
        )
        self.assertEqual(rows["run_sync"]["status"], "completed")
        self.assertEqual(rows["notify_ok"]["status"], "completed")
        self.assertEqual(
            rows["diagnose"]["status"],
            "skipped",
            "the branch jumped over diagnose; it must not have been walked into",
        )

    def test_the_record_says_what_it_compared(self):
        _task, rows = self._drive_auto_advance(
            _exit_code_flow(self.script["id"], "1")
        )
        record = (rows["check_exit"]["output"] or {})["condition"]
        self.assertEqual(record["matched_index"], 0)
        self.assertEqual(record["matched_label"], "실패")
        self.assertEqual(record["op"], "not_equals")
        self.assertEqual(record["resolved"], {"left": "1", "right": "0"})
        self.assertFalse(record["default"])
        self.assertEqual(record["evaluated"], [])

    def test_a_branch_event_is_appended_beside_the_goto(self):
        task, rows = self._drive_auto_advance(
            _exit_code_flow(self.script["id"], "1")
        )
        run_id = rows["check_exit"]["run_id"]
        events = self.store.list_events(run_id)
        kinds = [event.get("event_type") for event in events]
        self.assertIn("task.step.branch", kinds)
        self.assertIn("task.step.goto", kinds)
        branch = next(
            event for event in events if event.get("event_type") == "task.step.branch"
        )
        payload = branch.get("app_event") or {}
        self.assertEqual(payload.get("workflow_step_id"), "check_exit")
        self.assertEqual(payload.get("target_step_id"), "diagnose")
        self.assertEqual(payload.get("matched_index"), 0)
        self.assertFalse(payload.get("default"))
        self.assertEqual(payload.get("task_id"), task["id"])


class EvaluationFailureTest(ConditionRunnerTestCase):
    """No fallback: an unjudgeable predicate fails the step (spec 4.2)."""

    def _unresolvable_flow(self):
        return [
            {
                "id": "route",
                "type": "condition",
                "name": "재고 구간 분기",
                "branches": [
                    {
                        "label": "소량",
                        "when": {"left": "{{stock_count}}", "op": "lt", "right": "10"},
                        "target_step_id": "warn",
                    },
                    {"label": "충분", "when": None, "target_step_id": "notify_ok"},
                ],
            },
            {
                "id": "warn",
                "type": "notify",
                "name": "소량 경고",
                "notify": {"title": "재고 부족", "body": "10개 미만"},
                "on_success": {"type": "end"},
            },
            {
                "id": "notify_ok",
                "type": "notify",
                "name": "정상 알림",
                "notify": {"title": "재고 정상", "body": "충분"},
                "on_success": {"type": "end"},
            },
        ]

    def test_an_unbound_reference_does_not_fall_into_the_default_arm(self):
        # The default arm is "the remaining cases", not "judgement failed".
        # Sending an unreadable value there would leave the run record unable
        # to tell a healthy night from a blind one.
        task, _result = self._prepare(self._unresolvable_flow())
        asyncio.run(
            execute_task_orchestration(
                prepare_task_orchestration(
                    task["id"], provider_id="openai", auto_start=False
                )["execution"]
            )
        )
        rows = self._rows(task["id"])
        self.assertEqual(
            rows["notify_ok"]["status"],
            "queued",
            "the default arm must not absorb a failed judgement",
        )
        self.assertEqual(rows["warn"]["status"], "queued")

    def test_the_step_parks_on_a_person_with_a_message_naming_the_reference(self):
        task, result = self._prepare(self._unresolvable_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        rows = self._rows(task["id"])
        route = rows["route"]
        # `on_failure` for a condition defaults to ask_user, so the row parks
        # rather than staying `failed` — and the evaluation record survives
        # the park, because that record is what the person needs to read.
        self.assertEqual(route["status"], "waiting_for_user")
        record = (route.get("output") or {}).get("condition") or {}
        self.assertEqual(record.get("status"), "failed")
        self.assertEqual(record.get("reason"), "unbound_reference")
        self.assertEqual(record.get("branch_index"), 0)
        self.assertEqual(record.get("operand"), "left")
        self.assertIn("stock_count", record.get("message", ""))

    def test_the_same_failure_happens_on_the_single_step_path(self):
        task, _result = self._prepare(self._unresolvable_flow())
        row = self._rows(task["id"])["route"]
        outcome = asyncio.run(execute_task_step_adapter(task["id"], row["id"]))
        self.assertEqual(outcome["status"], "failed")
        record = (self._rows(task["id"])["route"].get("output") or {})["condition"]
        self.assertEqual(record["reason"], "unbound_reference")


class NoBranchesIsUntouchedTest(ConditionRunnerTestCase):
    """Every condition step saved before track H keeps its exact behaviour."""

    def _bare_flow(self):
        return [
            {"id": "decide", "type": "condition", "name": "Decide"},
            {
                "id": "after",
                "type": "notify",
                "name": "After",
                "notify": {"title": "done", "body": "done"},
                "on_success": {"type": "end"},
            },
        ]

    def test_it_still_emits_the_same_string_and_advances(self):
        task, result = self._prepare(self._bare_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        rows = self._rows(task["id"])
        self.assertEqual(
            rows["decide"]["output"],
            {"result": "condition step completed without branching"},
        )
        self.assertEqual(rows["after"]["status"], "completed")

    def test_the_single_step_path_emits_the_same_string(self):
        task, _result = self._prepare(self._bare_flow())
        row = self._rows(task["id"])["decide"]
        outcome = asyncio.run(execute_task_step_adapter(task["id"], row["id"]))
        self.assertEqual(
            outcome["output"],
            {"result": "condition step completed without branching"},
        )

    def test_no_branch_event_is_appended(self):
        task, result = self._prepare(self._bare_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        run_id = result["run"]["id"]
        kinds = [
            event.get("event_type") for event in self.store.list_events(run_id)
        ]
        self.assertNotIn("task.step.branch", kinds)


class BackwardArmTest(ConditionRunnerTestCase):
    """A backward arm is a loop now, and the budget is what ends it.

    Wave 3 measured the opposite and pinned it: `StepCursor.should_skip` walks
    past any row whose status is `completed`, so the second time round the
    condition row skipped *itself*, the target ran exactly once, and the run
    fell out the far end. Spec 4.3 carried that measurement as a finding for
    T-H-11.

    T-H-11 resolves it **without** loosening the completed-row rule — that
    rule is what keeps "already ran" meaning one thing for every step type.
    Instead the rows a backward arm walks back into are explicitly re-armed to
    `queued` by `_rearm_backward_branch`, which touches only *this run's* rows,
    only `completed` ones, and only inside `[target … condition]` of a
    backward **branch** arm. `completed` still means "already ran, do not run".

    So these assertions changed deliberately, which is what the wave-3
    docstring asked the day someone made a polling loop loop.
    """

    def _polling_flow(self):
        return [
            {
                "id": "poll",
                "type": "shell",
                "name": "작업 상태 조회",
                "script_id": self.script["id"],
                # Exits nonzero on purpose: this is the loop that never
                # terminates, and the budget is the only thing that ends it.
                "script_args": ["1"],
                "on_failure": {"type": "continue"},
                "on_success": {"type": "continue"},
            },
            {
                "id": "still_running",
                "type": "condition",
                "name": "아직 실행 중인가",
                "branches": [
                    {
                        "label": "아직 실행 중",
                        "when": {
                            "left": "{{steps.poll.stdout}}",
                            "op": "contains",
                            "right": "SYNC",
                        },
                        "target_step_id": "poll",
                    },
                    {"label": "끝났다", "when": None, "target_step_id": "report"},
                ],
            },
            {
                "id": "report",
                "type": "notify",
                "name": "결과 보고",
                "notify": {"title": "끝", "body": "끝"},
                "on_success": {"type": "end"},
            },
        ]

    def test_the_backward_arm_re_runs_its_target_until_the_budget_is_gone(self):
        # The predicate always holds (the script always prints SYNC OUTPUT),
        # so the arm the run takes is always the backward one. Before T-H-11
        # `poll` ran exactly twice and the run reported success; now it runs
        # until the transition budget is spent.
        task, result = self._prepare(self._polling_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        rows = self._rows(task["id"])
        record = (rows["still_running"].get("output") or {})["condition"]
        self.assertEqual(record["target_step_id"], "poll")
        self.assertFalse(record["default"], "the backward arm, not the default")

        events = [
            event.get("event_type")
            for event in self.store.list_events(result["run"]["id"])
        ]
        self.assertGreater(
            events.count("task.step.shell.started"),
            2,
            "the loop went round more than the single extra pass wave 3 measured",
        )
        self.assertEqual(
            events.count("task.step.branch"),
            events.count("task.step.shell.started"),
            "one branch decision per pass",
        )
        self.assertIn("task.step.branch.reentry", events)

    def test_a_re_armed_row_keeps_its_record_when_the_run_is_settled(self):
        """A looping row is `queued` at the end, and has run many times.

        Rows still queued on a finished run are settled as `skipped`
        (`_settle_steps_never_reached`), because `queued` on a run that is
        over is drawn by the app as *about to run*. A backward arm re-arms
        the rows it loops into, so this condition row ends the run queued
        having decided the branch on every pass — and the first version of
        that settlement read status alone, called it never-reached, and wrote
        its skip record straight over the `condition` decision underneath.

        Two things are checked because only together do they mean anything:
        the decision survives, and the row is not mislabelled as unreached.
        """
        task, result = self._prepare(self._polling_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        rows = self._rows(task["id"])
        looping = rows["still_running"]

        self.assertIn(
            "condition",
            looping.get("output") or {},
            "the branch decision was overwritten by the settlement",
        )
        self.assertNotIn(
            "skipped",
            looping.get("output") or {},
            "a row that ran on every pass was recorded as never reached",
        )
        # And the field that tells the two apart is `ended_at`, not
        # `started_at`. A condition row is decided in place and never goes
        # `running`, so it is never stamped as started — reading that field
        # alone called this row unreached on a run it decided every pass.
        self.assertIsNone(looping.get("started_at"))
        self.assertTrue(
            looping.get("ended_at"),
            "a condition row's only proof it ran is that it finished",
        )

    def test_a_loop_that_never_settles_fails_the_run_naming_the_arm(self):
        # The honest outcome for a poll that never concludes. Before T-H-11
        # this same workflow reported `completed` and sent the "끝" notify,
        # which is a green dot on a poll that never finished.
        task, result = self._prepare(self._polling_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        run = self.store.get_run(result["run"]["id"])
        self.assertEqual(run.get("status"), "failed")
        # The reason is written onto the *task*, not the run row —
        # `_finish_workflow_execution` sends the status to `update_run_status`
        # and the error to `update_task`.
        message = (
            (self.store.get_task(task["id"]).get("error") or {}).get("message") or ""
        )
        self.assertIn("transition limit exceeded", message)
        self.assertIn(
            "still_running",
            message,
            "budget_exhausted_message must name the arm that kept firing"
            " (T-H-06 built branch_history for exactly this)",
        )
        rows = self._rows(task["id"])
        self.assertEqual(
            rows["report"]["status"],
            "skipped",
            "the forward arm was never taken, so its row is settled, not queued",
        )

    def test_a_polling_loop_that_settles_leaves_through_the_forward_arm(self):
        # The shape the feature exists for: poll until the work is done, then
        # carry on. Impossible before T-H-11 — the condition row skipped
        # itself on the second pass.
        task, result = self._prepare(self._terminating_polling_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        rows = self._rows(task["id"])
        self.assertEqual(rows["poll"]["status"], "completed")
        self.assertEqual(rows["report"]["status"], "completed")
        self.assertEqual(
            self.counter.read_text().strip(),
            "3",
            "polled three times: RUNNING, RUNNING, DONE",
        )
        run = self.store.get_run(result["run"]["id"])
        self.assertEqual(run.get("status"), "completed")
        record = (rows["still_running"].get("output") or {})["condition"]
        self.assertEqual(
            record["target_step_id"], "report", "the last decision is the way out"
        )

    def test_no_row_is_left_saying_it_is_about_to_run(self):
        task, result = self._prepare(self._terminating_polling_flow())
        asyncio.run(execute_task_orchestration(result["execution"]))
        self.assertEqual(
            [
                row["status"]
                for row in self._rows(task["id"]).values()
                if row["status"] == "queued"
            ],
            [],
        )


class TheStaleRowIncidentIsStillRefusedTest(ConditionRunnerTestCase):
    """2026-08-06, replayed against a runner that now re-enters rows.

    A scheduled agent walked past every `completed` row on its task, landed on
    a `shell` row left `running` by a run three days earlier, ran it, and its
    `on_failure: goto_step` resolved to the wrong `diagnose` — 52 minutes of
    device script, ten times, over 8h45m (`_steps_for_run`).

    Making a backward arm loop means writing `queued` back onto rows that had
    finished, which is the one write that could put those rows back within
    reach. So the shape is replayed here: a first run finishes, one of its
    rows is left `running` the way the dead run left it, and a second run of
    the same task then loops through a backward arm. The earlier run's rows
    must come out untouched — not re-armed, not re-run, not even read.
    """

    def _snapshot(self, run_id):
        return {
            row["id"]: (row["status"], row.get("output"), row.get("updated_at"))
            for row in self.store.list_task_steps(self.task_id)
            if row.get("run_id") == run_id
        }

    def _run_once(self, task_id):
        result = prepare_task_orchestration(
            task_id, provider_id="openai", auto_start=False
        )
        assert result is not None
        asyncio.run(execute_task_orchestration(result["execution"]))
        return result["run"]["id"]

    def test_the_earlier_runs_rows_are_neither_re_armed_nor_re_run(self):
        task, first = self._prepare(self._terminating_polling_flow())
        self.task_id = task["id"]
        asyncio.run(execute_task_orchestration(first["execution"]))
        first_run_id = first["run"]["id"]
        first_polls = int(self.counter.read_text().strip())
        self.assertGreater(first_polls, 1, "the first run really did loop")

        # The dead run's signature: a shell row left `running`.
        stale = next(
            row
            for row in self.store.list_task_steps(task["id"])
            if row.get("run_id") == first_run_id
            and (row.get("input") or {}).get("workflow_step_id") == "poll"
        )
        self.store.update_task_step(stale["id"], {"status": "running"})
        before = self._snapshot(first_run_id)

        self.counter.write_text("0")
        second_run_id = self._run_once(task["id"])

        self.assertEqual(
            self._snapshot(first_run_id),
            before,
            "a backward arm in the second run must not touch the first run's rows",
        )
        self.assertEqual(
            self.store.get_task_step(stale["id"])["status"],
            "running",
            "the stale row is exactly where the dead run left it — not re-armed,"
            " not run, not settled",
        )
        self.assertEqual(
            int(self.counter.read_text().strip()),
            3,
            "the script ran the second run's own passes and not one more",
        )
        reentry = [
            event
            for event in self.store.list_events(second_run_id)
            if event.get("event_type") == "task.step.branch.reentry"
        ]
        self.assertTrue(reentry, "the second run did loop")
        rearmed = {
            step_id
            for event in reentry
            for step_id in (event.get("app_event") or {}).get("rearmed_step_ids", [])
        }
        self.assertEqual(rearmed, {"poll", "still_running"})
        self.assertNotIn(
            stale["id"],
            [
                (event.get("app_event") or {}).get("step_id")
                for event in self.store.list_events(second_run_id)
            ],
        )

    def test_the_dead_runs_shell_row_is_never_reached_at_all(self):
        # The incident in its own shape rather than by analogy: a task carrying
        # a `shell` row another run left `running`, pointing at a script that
        # records the fact it ran. A branching run — one that re-arms rows —
        # then fires on the same task.
        #
        # Measured, because the mechanism here is easy to mis-attribute: it is
        # `_steps_for_run`'s run scoping that keeps that row out of reach, not
        # `should_skip`'s completed-row rule. With the completed-row rule taken
        # out and run scoping left in, this scenario is still refused; with run
        # scoping taken out and the completed-row rule left in, the stale
        # script runs. So re-arming is gated on `run_id`, which is the guard
        # that actually holds.
        marker = self.dir / "dead_run_script_ran"
        danger = self.dir / "danger.sh"
        danger.write_text(f'#!/bin/bash\necho ran >> "{marker}"\nexit 0\n')
        danger.chmod(danger.stat().st_mode | stat.S_IEXEC)
        danger_script = script_store_module.get_script_store().register(
            name="dead_run_device_script", path=str(danger)
        )

        task, result = self._prepare(self._terminating_polling_flow())
        self.task_id = task["id"]
        self.store.create_task_step(
            task_id=task["id"],
            run_id="run_that_died_three_days_ago",
            capability_id=None,
            title="stale device script",
            status="running",
            input={
                "workflow_step_id": "device",
                "workflow_type": "shell",
                "script_id": danger_script["id"],
                "on_success": {"type": "continue"},
            },
        )
        asyncio.run(execute_task_orchestration(result["execution"]))

        self.assertFalse(
            marker.exists(),
            "the run reached a shell row a dead run left `running` and ran it —"
            " this is 2026-08-06",
        )
        stale = next(
            row
            for row in self.store.list_task_steps(task["id"])
            if row.get("run_id") == "run_that_died_three_days_ago"
        )
        self.assertEqual(
            stale["status"], "running", "not run, not re-armed, not settled"
        )

    def test_an_unstamped_task_re_arms_nothing_at_all(self):
        # The legacy hole `_steps_for_run` documents: when *no* row on the task
        # carries a run_id it falls back to the whole task's history. Re-arming
        # is gated on `run_id == this run`, so on that path it does nothing and
        # the walk behaves exactly as it did before T-H-11 — a backward arm
        # that does not loop is a great deal better than one that writes
        # `queued` over rows whose owner cannot be established.
        task, result = self._prepare(self._terminating_polling_flow())
        self.task_id = task["id"]
        for row in self.store.list_task_steps(task["id"]):
            self.store.update_task_step(row["id"], {"run_id": None})
        asyncio.run(execute_task_orchestration(result["execution"]))
        events = [
            event.get("event_type")
            for event in self.store.list_events(result["run"]["id"])
        ]
        self.assertNotIn("task.step.branch.reentry", events)
        self.assertEqual(
            self.counter.read_text().strip(),
            "1",
            "one pass and out: `poll` completed, so the second time round the"
            " walk skipped it and the condition row too — exactly what wave 3"
            " measured, and what this path still does",
        )


if __name__ == "__main__":
    unittest.main()
