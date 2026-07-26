"""A schedule has to survive an approval nobody answers.

``skip_if_active`` exists so two runs of the same task never overlap, and
``waiting_for_user`` used to count as active. Unattended there is nobody to
answer, so the run parked forever and every later firing was skipped — one
missed prompt killed the schedule permanently, quietly, with the only trace a
rising ``skip_count``.

A run that is *working* still blocks. A run that is *waiting on a human* blocks
only until the grace period, then gets abandoned so the schedule recovers.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import scheduler


def _stamp(seconds_ago: float) -> str:
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


class BlockingRunTest(unittest.TestCase):
    def _blocking(self, runs, *, grace: int = 3600):
        with patch.object(scheduler, "_runs_for_task", return_value=runs), patch.object(
            scheduler, "_stall_grace_seconds", return_value=grace
        ), patch.object(scheduler, "_abandon_stalled_run") as abandon:
            result = scheduler._blocking_run_for_task("task_1")
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
        self.assertEqual(reason, "previous run is waiting for approval")
        abandon.assert_not_called()

    def test_long_waiting_run_is_abandoned_so_the_schedule_recovers(self):
        run = {"id": "run_1", "status": "waiting_for_user", "updated_at": _stamp(7200)}
        (blocking, _), abandon = self._blocking([run])
        self.assertIsNone(blocking, "stalled run should not keep blocking")
        abandon.assert_called_once_with(run)

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
    def test_env_override(self):
        with patch.dict("os.environ", {"CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS": "90"}):
            self.assertEqual(scheduler._stall_grace_seconds(), 90)

    def test_garbage_env_falls_back_to_the_default(self):
        with patch.dict("os.environ", {"CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS": "soon"}):
            self.assertEqual(scheduler._stall_grace_seconds(), scheduler._STALL_GRACE_SECONDS)

    def test_zero_means_abandon_immediately(self):
        with patch.dict("os.environ", {"CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS": "0"}):
            self.assertEqual(scheduler._stall_grace_seconds(), 0)


if __name__ == "__main__":
    unittest.main()
