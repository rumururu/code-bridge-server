"""A schedule has to survive an approval nobody answers.

``skip_if_active`` exists so two runs of the same task never overlap, and
``waiting_for_user`` used to count as active. Unattended there is nobody to
answer, so the run parked forever and every later firing was skipped — one
missed prompt killed the schedule permanently, quietly, with the only trace a
rising ``skip_count``.

A run that is *working* still blocks. A run that is *waiting on a human* blocks
only until the grace period, then gets abandoned so the schedule recovers.

Two decisions, and since 2026-08-16 they live in two places, so this file is in
two halves:

* ``TaskScheduler._sweep_stalled_parks`` decides **whether to give up** on a
  parked run. It looks at every parked run on the tick, not just the ones a due
  schedule happens to point at — because a run with no schedule was otherwise
  never looked at by anything (see test_stalled_run_abandonment.py).
* ``_blocking_run_for_task`` decides **whether a schedule may fire**, and only
  that. It used to abandon runs itself; that copy of the policy is gone, since
  by the time it runs the sweep has already dealt with anything past the grace.

What abandonment *is* lives in test_stalled_run_abandonment.py: this file only
pins the decisions — which run blocks, which one gets given up on, and when.
"""

import asyncio
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import scheduler


def _stamp(seconds_ago: float) -> str:
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


class StalledParkSweepTest(unittest.TestCase):
    """Whether to give up on a parked run — the sweep's decision."""

    def _sweep(self, runs, *, grace: int = 3600, settling: bool = False):
        # `_abandon_stalled_run` is a coroutine: giving up goes through the deny
        # path, which has to await the orchestrator resume.
        with patch.object(scheduler, "_parked_runs", return_value=runs), patch.object(
            scheduler, "_stall_grace_seconds", return_value=grace
        ), patch.object(
            scheduler,
            "_abandon_stalled_run",
            AsyncMock(return_value={"abandoned": True, "path": "abandon"}),
        ) as abandon, patch(
            "agent.approval_resume.is_settling_run", return_value=settling
        ):
            asyncio.run(scheduler.TaskScheduler()._sweep_stalled_parks())
        return abandon

    def test_nothing_parked_abandons_nothing(self):
        self._sweep([]).assert_not_called()

    def test_a_recently_parked_run_is_left_alone(self):
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(60)}
        self._sweep([run]).assert_not_called()

    def test_a_long_parked_run_is_abandoned(self):
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(7200)}
        self._sweep([run]).assert_called_once_with(run)

    def test_a_run_already_being_settled_is_not_abandoned_again(self):
        # The approval expiry sweep runs earlier in the same tick and may
        # already have claimed this run. Abandoning it a second time would
        # drive one run down two paths at once.
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(7200)}
        self._sweep([run], settling=True).assert_not_called()

    def test_unparseable_timestamp_is_treated_as_stalled(self):
        # Better to let the run end than to leave it parked forever on a row we
        # cannot date.
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": "not-a-date"}
        self._sweep([run]).assert_called_once_with(run)

    def test_started_at_is_used_when_updated_at_is_missing(self):
        run = {"id": "run_1", "status": "waiting_for_user", "started_at": _stamp(60)}
        self._sweep([run]).assert_not_called()

    def test_grace_period_is_configurable(self):
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(120)}
        self._sweep([run], grace=60).assert_called_once_with(run)

    def test_one_bad_run_does_not_cost_the_tick(self):
        # Strictly best-effort: firing due schedules is the tick's job.
        with patch.object(
            scheduler, "_parked_runs", side_effect=RuntimeError("store is gone")
        ):
            asyncio.run(scheduler.TaskScheduler()._sweep_stalled_parks())


class ParkedRunsQueryTest(unittest.TestCase):
    """What the sweep reads, and what it refuses to read."""

    class _Store:
        def __init__(self, by_status):
            self.by_status = by_status
            self.calls: list[tuple[str, int]] = []

        def list_runs(self, *, status=None, limit=50):
            self.calls.append((status, limit))
            return list(self.by_status.get(status, []))

    def _parked(self, store):
        with patch.object(scheduler, "get_agent_store", lambda: store):
            return scheduler._parked_runs()

    def test_only_parked_statuses_are_queried_and_the_read_is_capped(self):
        store = self._Store(
            {
                "waiting_for_user": [{"id": "run_1", "task_id": "task_1"}],
                "blocked": [{"id": "run_2", "task_id": "task_1"}],
            }
        )
        runs = self._parked(store)

        self.assertEqual({run["id"] for run in runs}, {"run_1", "run_2"})
        # Three indexed seeks, one per parked status — never a scan of the
        # whole run history. `completed`/`failed` are never asked for.
        self.assertEqual(
            {status for status, _ in store.calls},
            scheduler._WAITING_RUN_STATUSES,
        )
        self.assertEqual(
            {limit for _, limit in store.calls}, {scheduler._PARK_SWEEP_LIMIT}
        )

    def test_a_run_with_no_task_is_skipped(self):
        # Abandonment is task-shaped; retrying a run it cannot end would log a
        # warning every 30s forever.
        store = self._Store({"waiting_for_user": [{"id": "run_1"}]})
        self.assertEqual(self._parked(store), [])

    def test_a_run_in_two_statuses_is_returned_once(self):
        row = {"id": "run_1", "task_id": "task_1"}
        store = self._Store({"waiting_for_user": [row], "waiting_user": [row]})
        self.assertEqual(len(self._parked(store)), 1)


class BlockingRunTest(unittest.TestCase):
    """Whether a schedule may fire — a report, and nothing more."""

    def _blocking(self, runs, *, settling: bool = False):
        with patch.object(
            scheduler, "_runs_for_task", return_value=runs
        ), patch("agent.approval_resume.is_settling_run", return_value=settling):
            return scheduler._blocking_run_for_task("task_1")

    def test_no_runs_does_not_block(self):
        blocking, _ = self._blocking([])
        self.assertIsNone(blocking)

    def test_finished_run_does_not_block(self):
        blocking, _ = self._blocking(
            [{"id": "run_1", "status": "completed", "updated_at": _stamp(10)}]
        )
        self.assertIsNone(blocking)

    def test_running_run_blocks_regardless_of_age(self):
        # Long-running is not stalled — a 4-hour build must not be duplicated.
        blocking, reason = self._blocking(
            [{"id": "run_1", "status": "running", "updated_at": _stamp(14400)}]
        )
        self.assertEqual(blocking["id"], "run_1")
        self.assertEqual(reason, "previous run still active")

    def test_a_parked_run_blocks_with_its_own_reason(self):
        blocking, reason = self._blocking(
            [{"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(60)}]
        )
        self.assertEqual(blocking["id"], "run_1")
        # "a person", not "approval": the same branch covers an `ask_user`
        # park, which has no approval in it to wait for.
        self.assertEqual(reason, "previous run is waiting for a person")

    def test_a_run_being_settled_says_so(self):
        # Settling through the deny path is asynchronous, so the run has not
        # landed anywhere yet and firing over it would stack two runs.
        blocking, reason = self._blocking(
            [{"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(7200)}],
            settling=True,
        )
        self.assertEqual(blocking["id"], "run_1")
        self.assertEqual(reason, "previous run is being settled")

    def test_it_never_abandons_anything_itself(self):
        # The give-up policy lives in the sweep now. A second copy here could
        # only ever disagree with it.
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(99999)}
        with patch.object(scheduler, "_abandon_stalled_run") as abandon:
            blocking, _ = self._blocking([run])
        abandon.assert_not_called()
        self.assertEqual(blocking["id"], "run_1")

    def test_progressing_run_wins_over_a_stalled_sibling(self):
        runs = [
            {"id": "run_old", "status": "waiting_for_user", "updated_at": _stamp(7200)},
            {"id": "run_new", "status": "running", "updated_at": _stamp(5)},
        ]
        blocking, _ = self._blocking(runs)
        self.assertEqual(blocking["id"], "run_new")


class GraceConfigTest(unittest.TestCase):
    """One deadline by default; a second one only if somebody asks for it.

    There used to be two independent answers to "nobody answered this" — this
    grace (1h) and the approval deadline (24h) — and because the shorter always
    wins, the 1h default silently pre-empted every approval deadline on a
    scheduled run. Unset, the grace now *is* the approval deadline.
    """

    def test_env_override(self):
        with patch.dict("os.environ", {"CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS": "90"}):
            self.assertEqual(scheduler._stall_grace_seconds(), 90)

    def test_unset_inherits_the_approval_deadline(self):
        env = {"CODEBRIDGE_APPROVAL_EXPIRY_SECONDS": "1234"}
        with patch.dict("os.environ", env, clear=False):
            import os

            os.environ.pop("CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS", None)
            self.assertEqual(scheduler._stall_grace_seconds(), 1234)

    def test_garbage_env_falls_back_to_the_inherited_deadline(self):
        # Migrated: this used to assert the standalone 1h default. Garbage is
        # "no answer given", which now means the approval deadline answers.
        env = {
            "CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS": "soon",
            "CODEBRIDGE_APPROVAL_EXPIRY_SECONDS": "4321",
        }
        with patch.dict("os.environ", env, clear=False):
            self.assertEqual(scheduler._stall_grace_seconds(), 4321)

    def test_approvals_that_never_expire_still_leave_a_backstop(self):
        # `CODEBRIDGE_APPROVAL_EXPIRY_SECONDS=0` means approvals never expire,
        # so nothing would ever settle a parked run and the schedule would be
        # wedged for good. The standing grace is what stops that.
        env = {"CODEBRIDGE_APPROVAL_EXPIRY_SECONDS": "0"}
        with patch.dict("os.environ", env, clear=False):
            import os

            os.environ.pop("CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS", None)
            self.assertEqual(
                scheduler._stall_grace_seconds(), scheduler._STALL_GRACE_SECONDS
            )

    def test_zero_means_abandon_immediately(self):
        with patch.dict("os.environ", {"CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS": "0"}):
            self.assertEqual(scheduler._stall_grace_seconds(), 0)


if __name__ == "__main__":
    unittest.main()
