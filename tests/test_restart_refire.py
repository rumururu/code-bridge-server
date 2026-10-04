"""A restart re-fires the schedules whose runs it interrupted.

``reconcile_interrupted_runs`` closes a run the previous process left
mid-flight so ``skip_if_active`` stops skipping — but closing it is not
running it. A deploy during the 08:00 morning check left that day with no
check, its notification never sent; a 20-minute device cycle was killed at
minute 19 and lost. Schedules owning an interrupted run fire once at start.
Manual runs (no schedule) stay closed: re-running one unasked is a surprise.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from agent.run_reconciliation import reconcile_interrupted_runs  # noqa: E402
from agent.scheduler import refire_after_shutdown  # noqa: E402
from core import database  # noqa: E402


class RestartRefireTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "refire.db"
        agent_store._agent_store = None
        schedule_store._schedule_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = agent_store.get_agent_store()
        self.schedules = schedule_store.get_schedule_store()

    def _restore(self) -> None:
        agent_store._agent_store = None
        schedule_store._schedule_store = None
        database.DB_PATH = self._original

    def _scheduled_task(self, *, enabled=True):
        task = self.store.create_task(title="Morning check", kind="ops")
        schedule = self.schedules.create(
            task_id=task["id"], expression={"kind": "interval", "seconds": 3600}, enabled=enabled
        )
        return task, schedule

    def _interrupted_run(self, task):
        run = self.store.create_run(task_id=task["id"], title="Run task: Morning check")
        self.store.update_run_status(run["id"], "running")
        return run

    async def test_the_schedule_of_an_interrupted_run_fires_once(self):
        task, schedule = self._scheduled_task()
        self._interrupted_run(task)
        closed = reconcile_interrupted_runs()
        fire = AsyncMock()

        fired = await refire_after_shutdown(closed, fire=fire)

        self.assertEqual([s["id"] for s in fired], [schedule["id"]])
        fire.assert_awaited_once()
        self.assertEqual(fire.await_args.args[0]["id"], schedule["id"])

    async def test_restart_records_both_run_links_without_marking_success(self):
        from agent.experience_service import summary
        task, schedule = self._scheduled_task()
        old = self._interrupted_run(task)
        closed = reconcile_interrupted_runs()
        replacement = self.store.create_run(task_id=task["id"], title="Restart")
        await refire_after_shutdown(closed, fire=AsyncMock(return_value=replacement["id"]))
        self.assertEqual(summary(old["id"])["related_runs"][0]["run_id"], replacement["id"])
        self.assertEqual(summary(replacement["id"])["related_runs"][0]["run_id"], old["id"])
        self.assertEqual(summary(replacement["id"])["verification"]["status"], "unknown")

    async def test_two_interrupted_runs_of_one_task_fire_the_schedule_once(self):
        task, schedule = self._scheduled_task()
        self._interrupted_run(task)
        self._interrupted_run(task)
        closed = reconcile_interrupted_runs()
        fire = AsyncMock()

        fired = await refire_after_shutdown(closed, fire=fire)

        self.assertEqual(len(fired), 1)
        fire.assert_awaited_once()

    async def test_a_manual_run_is_closed_but_not_re_run(self):
        task = self.store.create_task(title="One-off", kind="ops")
        self._interrupted_run(task)
        closed = reconcile_interrupted_runs()
        fire = AsyncMock()

        fired = await refire_after_shutdown(closed, fire=fire)

        self.assertEqual(fired, [])
        fire.assert_not_awaited()

    async def test_a_disabled_schedule_is_not_revived(self):
        task, _ = self._scheduled_task(enabled=False)
        self._interrupted_run(task)
        closed = reconcile_interrupted_runs()
        fire = AsyncMock()

        self.assertEqual(await refire_after_shutdown(closed, fire=fire), [])
        fire.assert_not_awaited()

    async def test_nothing_interrupted_fires_nothing(self):
        fire = AsyncMock()
        self.assertEqual(await refire_after_shutdown([], fire=fire), [])
        fire.assert_not_awaited()

    async def test_one_failing_fire_does_not_stop_the_rest(self):
        task_a, schedule_a = self._scheduled_task()
        task_b, schedule_b = self._scheduled_task()
        self._interrupted_run(task_a)
        self._interrupted_run(task_b)
        closed = reconcile_interrupted_runs()

        async def fire(schedule):
            if schedule["id"] == schedule_a["id"]:
                raise RuntimeError("orchestrator down")

        fired = await refire_after_shutdown(closed, fire=AsyncMock(side_effect=fire))
        self.assertEqual([s["id"] for s in fired], [schedule_b["id"]])
