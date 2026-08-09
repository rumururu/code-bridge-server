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

This is also the only periodic timer the server owns, so other work that has
to happen "every so often" rides this tick rather than starting a rival loop —
see :meth:`TaskScheduler._sweep_subagents_if_due`, which asks
:mod:`agent.subagent_sweep` whether its (much coarser) interval has elapsed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from agent.agent_store import get_agent_store
from agent.schedule_store import get_schedule_store
from agent.task_orchestrator import (
    execute_task_orchestration,
    prepare_task_orchestration,
)

logger = logging.getLogger(__name__)


# A run in one of these is doing work; firing again would stack two runs on
# the same task.
_PROGRESSING_RUN_STATUSES = {
    "queued",
    "starting",
    "running",
}

# A run in one of these has stopped and is waiting on a human. Unattended, that
# human never arrives: the run sits there forever and — because it counts as
# "active" — silently swallows every later firing of the schedule. One
# unanswered approval used to kill a schedule permanently.
_WAITING_RUN_STATUSES = {
    "blocked",
    "waiting_for_user",
    "waiting_user",
}

_ACTIVE_RUN_STATUSES = _PROGRESSING_RUN_STATUSES | _WAITING_RUN_STATUSES

# How long a run may sit waiting on a human before the scheduler gives up on
# it and lets the schedule move on. Long enough that someone glancing at their
# phone within the hour still gets to answer; short enough that a schedule
# recovers on its own overnight.
_STALL_GRACE_SECONDS = 3600


def _stall_grace_seconds() -> int:
    raw = os.environ.get("CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS")
    if raw is None:
        return _STALL_GRACE_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _STALL_GRACE_SECONDS
    return max(0, value)


def _runs_for_task(task_id: str) -> list[dict[str, Any]]:
    try:
        store = get_agent_store()
    except Exception:
        return []
    try:
        return store.list_runs(task_id=task_id, limit=5)
    except TypeError:
        # Older signature without task_id filter; fall back to status check.
        runs = store.list_runs(limit=20)
        return [run for run in runs if run.get("task_id") == task_id]
    except Exception:
        logger.exception("scheduler: failed to inspect task runs")
        return []


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    # Stored timestamps are naive UTC (SQLite CURRENT_TIMESTAMP).
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _waiting_seconds(run: dict[str, Any]) -> float | None:
    """How long ``run`` has been parked, or ``None`` if that can't be told."""
    stamp = _parse_timestamp(run.get("updated_at")) or _parse_timestamp(run.get("started_at"))
    if stamp is None:
        return None
    return (datetime.now(timezone.utc) - stamp).total_seconds()


def _abandon_stalled_run(run: dict[str, Any]) -> None:
    """Fail a run nobody answered, and expire the approvals holding it.

    Without this the run stays "active" forever. Expiring its approvals also
    keeps the pending queue honest — those prompts are about a run that is no
    longer going anywhere.
    """
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        return
    try:
        from approvals.approval_store import get_approval_store

        approvals = get_approval_store()
        for approval in approvals.list_pending(run_id=run_id):
            approvals.mark_expired(approval["id"])
    except Exception:
        logger.exception("scheduler: failed to expire approvals for run %s", run_id)
    try:
        get_agent_store().update_run_status(run_id, "failed")
    except Exception:
        logger.exception("scheduler: failed to abandon stalled run %s", run_id)


def _blocking_run_for_task(task_id: str) -> tuple[dict[str, Any] | None, str]:
    """The run that should stop this firing, and why.

    Returns ``(None, "")`` when the schedule is free to fire. A run waiting on
    a human past the grace period is abandoned here rather than reported as
    blocking, which is what lets a schedule recover by itself.
    """
    runs = _runs_for_task(task_id)
    waiting: list[dict[str, Any]] = []
    for run in runs:
        status = run.get("status")
        if status in _PROGRESSING_RUN_STATUSES:
            return run, "previous run still active"
        if status in _WAITING_RUN_STATUSES:
            waiting.append(run)

    grace = _stall_grace_seconds()
    for run in waiting:
        elapsed = _waiting_seconds(run)
        if elapsed is not None and elapsed < grace:
            return run, "previous run is waiting for approval"

    for run in waiting:
        logger.warning(
            "scheduler: abandoning run %s — waiting on a human for longer than %ss",
            run.get("id"),
            grace,
        )
        _abandon_stalled_run(run)
    return None, ""


def _has_active_run_for_task(task_id: str) -> bool:
    """Return True if any non-terminal run still exists for the task."""
    return any(run.get("status") in _ACTIVE_RUN_STATUSES for run in _runs_for_task(task_id))


async def _fire_schedule(schedule: dict[str, Any]) -> None:
    schedule_id = schedule["id"]
    task_id = schedule["task_id"]
    store = get_schedule_store()

    if schedule.get("skip_if_active"):
        blocking, reason = _blocking_run_for_task(task_id)
        if blocking is not None:
            logger.info(
                "scheduler: skipping schedule %s — task %s: %s",
                schedule_id,
                task_id,
                reason,
            )
            store.record_fire(
                schedule_id,
                run_id=None,
                status="skipped",
                error=reason,
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
        await self._sweep_subagents_if_due()
        return len(due)

    async def _sweep_subagents_if_due(self) -> None:
        """Let the subagent sweep piggyback on this loop, if it is due.

        Deliberately not a second background task: this process already owns
        exactly one periodic timer, and a second one would mean two things to
        start, two to stop, and two ways for the lifespan to leak a task. The
        sweep's own cadence (hours, see :mod:`agent.subagent_sweep`) is far
        coarser than this tick, so it decides for itself whether it is due
        from a persisted timestamp and almost always answers no.

        Strictly best-effort and strictly last: firing due schedules is this
        loop's job, and a filesystem walk failing must never cost a scheduled
        run.
        """
        try:
            from agent.subagent_sweep import maybe_run_due_subagent_sweep

            await asyncio.to_thread(maybe_run_due_subagent_sweep)
        except Exception:
            logger.exception("scheduler: subagent sweep failed")


_scheduler: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
