"""A schedule has to survive an approval nobody answers.

``skip_if_active`` exists so two runs of the same task never overlap, and
``waiting_for_user`` used to count as active. Unattended there is nobody to
answer, so the run parked forever and every later firing was skipped — one
missed prompt killed the schedule permanently, quietly, with the only trace a
rising ``skip_count``.

A run that is *working* still blocks. A run that is *waiting on a human* blocks
only until the grace period, then gets abandoned so the schedule recovers.

What abandonment *is* lives in test_stalled_run_abandonment.py: this file only
pins the decision — which run blocks, which one gets given up on, and when.
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


class BlockingRunTest(unittest.TestCase):
    def _blocking(self, runs, *, grace: int = 3600, abandon_path: str = "abandon"):
        # `_abandon_stalled_run` is a coroutine now: giving up on a run goes
        # through the deny path, which has to await the orchestrator resume.
        with patch.object(scheduler, "_runs_for_task", return_value=runs), patch.object(
            scheduler, "_stall_grace_seconds", return_value=grace
        ), patch.object(
            scheduler,
            "_abandon_stalled_run",
            AsyncMock(return_value={"abandoned": True, "path": abandon_path}),
        ) as abandon:
            result = asyncio.run(scheduler._blocking_run_for_task("task_1"))
        return result, abandon

    def test_no_runs_does_not_block(self):
        (blocking, reason), abandon = self._blocking([])
        self.assertIsNone(blocking)
        abandon.assert_not_called()

    def test_finished_run_does_not_block(self):
        (blocking, _), _ = self._blocking(
            [{"id": "run_1", "status": "completed", "updated_at": _stamp(10)}]
        )
        self.assertIsNone(blocking)

    def test_running_run_blocks_regardless_of_age(self):
        # Long-running is not stalled — a 4-hour build must not be duplicated.
        (blocking, reason), abandon = self._blocking(
            [{"id": "run_1", "status": "running", "updated_at": _stamp(14400)}]
        )
        self.assertEqual(blocking["id"], "run_1")
        self.assertEqual(reason, "previous run still active")
        abandon.assert_not_called()

    def test_recently_waiting_run_blocks_with_its_own_reason(self):
        (blocking, reason), abandon = self._blocking(
            [{"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(60)}]
        )
        self.assertEqual(blocking["id"], "run_1")
        # "a person", not "approval": the same branch covers an `ask_user`
        # park, which has no approval in it to wait for.
        self.assertEqual(reason, "previous run is waiting for a person")
        abandon.assert_not_called()

    def test_long_waiting_run_is_abandoned_so_the_schedule_recovers(self):
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(7200)}
        (blocking, _), abandon = self._blocking([run])
        self.assertIsNone(blocking, "stalled run should not keep blocking")
        abandon.assert_called_once_with(run)

    def test_a_run_settled_down_the_deny_path_blocks_until_the_next_tick(self):
        # Denying an unanswered approval hands the parked provider turn a
        # refusal and lets the step's failure policy run — asynchronously. The
        # run has not landed anywhere yet, so firing now would stack a second
        # run on top of one still being wound down.
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(7200)}
        (blocking, reason), abandon = self._blocking([run], abandon_path="deny")
        self.assertEqual(blocking["id"], "run_1")
        self.assertEqual(reason, "previous run is being settled")
        abandon.assert_called_once_with(run)

    def test_a_run_already_being_settled_is_not_abandoned_again(self):
        # The expiry sweep runs earlier in the same tick and may already have
        # claimed this run. Abandoning it a second time would drive one run
        # down two paths at once.
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(7200)}
        with patch("agent.approval_resume.is_settling_run", return_value=True):
            (blocking, reason), abandon = self._blocking([run])
        self.assertEqual(blocking["id"], "run_1")
        self.assertEqual(reason, "previous run is being settled")
        abandon.assert_not_called()

    def test_progressing_run_wins_over_a_stalled_sibling(self):
        runs = [
            {"id": "run_old", "status": "waiting_for_user", "updated_at": _stamp(7200)},
            {"id": "run_new", "status": "running", "updated_at": _stamp(5)},
        ]
        (blocking, _), abandon = self._blocking(runs)
        self.assertEqual(blocking["id"], "run_new")
        abandon.assert_not_called()

    def test_unparseable_timestamp_is_treated_as_stalled(self):
        # Better to let the schedule recover than to wedge it forever on a row
        # we cannot date.
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": "not-a-date"}
        (blocking, _), abandon = self._blocking([run])
        self.assertIsNone(blocking)
        abandon.assert_called_once_with(run)

    def test_started_at_is_used_when_updated_at_is_missing(self):
        run = {"id": "run_1", "status": "waiting_for_user", "started_at": _stamp(60)}
        (blocking, _), _ = self._blocking([run])
        self.assertEqual(blocking["id"], "run_1")

    def test_grace_period_is_configurable(self):
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(120)}
        (blocking, _), _ = self._blocking([run], grace=60)
        self.assertIsNone(blocking)


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
