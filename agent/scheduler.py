"""Background scheduler that fires due task_schedules.

This is the unattended-execution layer. It does **not** bypass the policy /
approval / audit boundary — each firing goes through ``prepare_task_orchestration``
followed by ``execute_task_orchestration``, the same path a manual
``POST /api/agent/tasks/{task_id}/start`` call uses. Approvals raised by
policy_engine during step execution stack up in the pending approvals queue
exactly as they would for an interactive run.

The loop is intentionally simple: poll every ``tick_seconds`` (default 30s),
find ``next_run_at <= now``, fire one at a time, record the outcome, and
advance ``next_run_at`` based on the schedule expression.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.agent_store import get_agent_store
from agent.schedule_store import get_schedule_store
from agent.task_orchestrator import (
    execute_task_orchestration,
    prepare_task_orchestration,
)

logger = logging.getLogger(__name__)


_ACTIVE_RUN_STATUSES = {
    "queued",
    "starting",
    "running",
    "blocked",
    "waiting_for_user",
    "waiting_user",
}


def _has_active_run_for_task(task_id: str) -> bool:
    """Return True if any non-terminal run still exists for the task."""
    try:
        store = get_agent_store()
    except Exception:
        return False
    try:
        runs = store.list_runs(task_id=task_id, limit=5)
    except TypeError:
        # Older signature without task_id filter; fall back to status check.
        runs = store.list_runs(limit=20)
        runs = [run for run in runs if run.get("task_id") == task_id]
    except Exception:
        logger.exception("scheduler: failed to inspect task runs")
        return False
    for run in runs:
        if run.get("status") in _ACTIVE_RUN_STATUSES:
            return True
    return False


async def _fire_schedule(schedule: dict[str, Any]) -> None:
    schedule_id = schedule["id"]
    task_id = schedule["task_id"]
    store = get_schedule_store()

    if schedule.get("skip_if_active") and _has_active_run_for_task(task_id):
        logger.info(
            "scheduler: skipping schedule %s — task %s still has an active run",
            schedule_id,
            task_id,
        )
        store.record_fire(
            schedule_id,
            run_id=None,
            status="skipped",
            error="previous run still active",
        )
        return

    try:
        prepared = await asyncio.to_thread(
            prepare_task_orchestration,
            task_id,
            provider_id=schedule.get("provider_id"),
            model=schedule.get("model"),
            cwd=schedule.get("cwd"),
            prompt=schedule.get("prompt"),
            requested_capabilities=schedule.get("capabilities") or None,
            auto_start=True,
            dry_run=False,
        )
    except ValueError as exc:
        logger.warning(
            "scheduler: schedule %s rejected by orchestrator: %s",
            schedule_id,
            exc,
        )
        store.record_fire(schedule_id, run_id=None, status="error", error=str(exc))
        return
    except Exception as exc:
        logger.exception("scheduler: unexpected orchestrator failure")
        store.record_fire(schedule_id, run_id=None, status="error", error=str(exc))
        return

    if not prepared:
        logger.warning(
            "scheduler: schedule %s task %s not found, disabling",
            schedule_id,
            task_id,
        )
        store.record_fire(
            schedule_id,
            run_id=None,
            status="error",
            error="task not found",
        )
        store.update(schedule_id, {"enabled": False})
        return

    execution = prepared.get("execution")
    run = prepared.get("run") or {}
    run_id = run.get("id")
    store.record_fire(schedule_id, run_id=run_id, status="fired")

    if isinstance(execution, dict):
        try:
            asyncio.create_task(execute_task_orchestration(execution))
        except Exception:
            logger.exception(
                "scheduler: failed to spawn execution for run %s", run_id
            )


class TaskScheduler:
    """Singleton background loop. Owned by the FastAPI lifespan."""

    def __init__(self, *, tick_seconds: float = 30.0) -> None:
        self._tick = max(5.0, float(tick_seconds))
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="task-scheduler")
        logger.info("scheduler: started (tick=%.1fs)", self._tick)

    async def stop(self) -> None:
        if not self.running:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=self._tick + 5.0)
        except asyncio.TimeoutError:
            if self._task is not None:
                self._task.cancel()
        finally:
            self._task = None
            self._stop_event = None
        logger.info("scheduler: stopped")

    async def trigger_once(self) -> int:
        """Run one tick immediately (used by manual trigger / tests). Returns fired count."""
        return await self._tick_once()

    async def _run(self) -> None:
        assert self._stop_event is not None
        # First tick after a short delay so app startup finishes first.
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
            return
        except asyncio.TimeoutError:
            pass
        while not self._stop_event.is_set():
            try:
                await self._tick_once()
            except Exception:
                logger.exception("scheduler: tick failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._tick,
                )
            except asyncio.TimeoutError:
                continue

    async def _tick_once(self) -> int:
        store = get_schedule_store()
        due = await asyncio.to_thread(store.list_due)
        for schedule in due:
            await _fire_schedule(schedule)
        return len(due)


_scheduler: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
