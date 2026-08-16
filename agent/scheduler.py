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
see :meth:`TaskScheduler._sweep_cli_agents_if_due`, which asks
:mod:`agent.cli_agent_sweep` whether its (much coarser) interval has elapsed,
:meth:`TaskScheduler._sweep_expired_approvals`, which ends approvals whose
``expires_at`` has passed so the runs parked on them stop waiting forever, and
:meth:`TaskScheduler._sweep_stalled_parks`, which ends the parks that have no
approval behind them at all — the ones no schedule would ever have looked at.
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

# Last-resort grace, used only when there is no approval deadline to inherit
# (expiry switched off). Long enough that someone glancing at their phone
# within the hour still gets to answer; short enough that a schedule recovers
# on its own overnight.
_STALL_GRACE_SECONDS = 3600

# Most parked runs the park sweep will read per status in one tick. Not a
# throttle on abandonment — anything beyond it is picked up on a later tick —
# but a ceiling on what one tick can cost if a store ever holds an absurd
# number of simultaneously parked runs.
_PARK_SWEEP_LIMIT = 200


def _stall_grace_seconds() -> int:
    """How long a park may hold a schedule before the scheduler gives up.

    There used to be two independent answers to "nobody answered this": this
    grace (1h) and the approval deadline (``CODEBRIDGE_APPROVAL_EXPIRY_SECONDS``,
    24h). Since the shorter one always wins, the 1h default silently overrode
    every approval deadline for scheduled runs, and the expiry sweep — the one
    path that settles the parked provider turn — almost never got to act.

    So there is one number by default: unset, this *is* the approval deadline.
    Setting ``CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS`` is an explicit "hold my
    schedule for no longer than this, whatever the approval says", which is a
    real thing to want for a schedule that has to keep cadence — and it is now
    a decision someone made rather than a default nobody knew about. Either
    way, what happens *when* the grace runs out is identical: the run is
    abandoned through :func:`agent.approval_resume.abandon_waiting_run`, which
    is the same door the expiry sweep uses.
    """
    raw = os.environ.get("CODEBRIDGE_SCHEDULE_STALL_GRACE_SECONDS")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    try:
        from approvals.approval_service import approval_expiry_seconds

        inherited = approval_expiry_seconds()
    except Exception:
        logger.exception("scheduler: could not read the approval expiry window")
        inherited = 0
    # ``0`` means approvals never expire, so nothing would ever settle a parked
    # run and the schedule would be wedged for good. Fall back to the standing
    # grace: a schedule that can never recover is the worse failure.
    return inherited if inherited > 0 else _STALL_GRACE_SECONDS


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


async def _abandon_stalled_run(run: dict[str, Any]) -> dict[str, Any] | None:
    """Give up on a run nobody answered — through the one shared path.

    Everything this used to do itself (expire the approvals, write ``failed``
    onto the run row) now lives in :func:`agent.approval_resume.abandon_waiting_run`,
    which is also what an approval timing out on its own goes through. The
    scheduler still decides *whether* to give up; it no longer decides what
    giving up looks like, because its own version skipped the deny path and
    left the provider session holding an unanswered permission callback.
    """
    try:
        from agent.approval_resume import abandon_waiting_run

        return await abandon_waiting_run(run, waited_seconds=_waiting_seconds(run))
    except Exception:
        logger.exception("scheduler: failed to abandon stalled run %s", run.get("id"))
        return None


def _parked_runs() -> list[dict[str, Any]]:
    """Every run currently stopped waiting for a person, across all tasks.

    Deliberately queried by *status* rather than by walking tasks or schedules.
    ``agent_runs`` is indexed on ``status`` (``idx_agent_runs_status``,
    core/database.py:150), and the statuses asked for here are the parked ones,
    so the work is proportional to how many runs are parked *right now* — a
    handful — and not to how many runs the store has ever held. A year of
    finished runs costs nothing, because ``completed``/``failed`` rows are never
    read.

    Capped per status, so a store that somehow accumulated thousands of parked
    runs cannot turn one tick into an unbounded read. The cap sorts newest
    first, so in that (unreached) case the oldest parks wait for a later tick
    once the newer ones have been settled and left the set.
    """
    try:
        store = get_agent_store()
    except Exception:
        return []
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for status in sorted(_WAITING_RUN_STATUSES):
        try:
            rows = store.list_runs(status=status, limit=_PARK_SWEEP_LIMIT)
        except TypeError:
            # Older store signature without a status filter. Rather than read
            # every run to emulate it, leave the parked runs to the per-task
            # check; a wrong-shaped store must not make this loop expensive.
            logger.warning("scheduler: run store cannot filter by status")
            return []
        except Exception:
            logger.exception("scheduler: failed to list %s runs", status)
            continue
        for row in rows:
            run_id = row.get("id")
            if not isinstance(run_id, str) or not run_id or run_id in seen:
                continue
            if not row.get("task_id"):
                # Abandonment is task-shaped (it fails the parked step and
                # closes the task out). A run with no task has no park this
                # module knows how to end, and retrying it every 30s would log
                # a warning a minute forever.
                continue
            seen.add(run_id)
            runs.append(row)
    return runs


def _blocking_run_for_task(task_id: str) -> tuple[dict[str, Any] | None, str]:
    """The run that should stop this firing, and why.

    Returns ``(None, "")`` when the schedule is free to fire. Purely a report:
    this used to abandon a stalled run itself, but
    :meth:`TaskScheduler._sweep_stalled_parks` now does that for *every* parked
    run earlier in the same tick, schedule or no schedule. Anything still
    waiting when this runs is either inside the grace period or is being
    settled right now, and both of those block rather than get abandoned —
    so a second copy of the give-up policy here would have nothing left to do
    but disagree with the first.
    """
    runs = _runs_for_task(task_id)
    waiting: list[dict[str, Any]] = []
    for run in runs:
        status = run.get("status")
        if status in _PROGRESSING_RUN_STATUSES:
            return run, "previous run still active"
        if status in _WAITING_RUN_STATUSES:
            waiting.append(run)

    from agent.approval_resume import is_settling_run

    for run in waiting:
        if is_settling_run(run.get("id")):
            # Being wound down right now — by a decision from the app, by the
            # approval expiry sweep, or by the park sweep earlier in this tick.
            # Settling through the deny path is asynchronous (it answers the
            # parked provider turn and then applies the step's failure policy),
            # so the run has not landed anywhere yet and firing over the top of
            # it would stack two runs on one task. Reported ahead of a plain
            # park because it is the more specific answer to "why not now".
            return run, "previous run is being settled"
    if waiting:
        return waiting[0], "previous run is waiting for a person"
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
        # Sweep *before* firing, not after. Both halves of this tick can decide
        # to end the same parked run — the sweep because its approval deadline
        # passed, the schedule check because its stall grace ran out — and by
        # default those two deadlines are now the same number, so they land on
        # the same tick. Sweeping first means the sweep has already claimed the
        # run (see `agent.approval_resume.is_settling_run`) by the time
        # `_blocking_run_for_task` looks at it, so the run is settled once and
        # the schedule fires on the next tick instead of over the top of a run
        # still being wound down.
        await self._sweep_expired_approvals()
        await self._sweep_stalled_parks()
        for schedule in due:
            await _fire_schedule(schedule)
        await self._sweep_cli_agents_if_due()
        return len(due)

    async def _sweep_stalled_parks(self) -> None:
        """End every park nobody answered, schedule or no schedule.

        Giving up on a stalled run used to happen in exactly one place:
        ``_blocking_run_for_task``, which only ever runs when a schedule of
        that task is due. A run with no schedule — a ``run once`` agent, a task
        whose schedule was disabled, anything started by hand — was therefore
        never looked at, and an unanswered ``ask_user`` park on one of those sat
        in ``waiting_for_user`` for good, piling up in the user's run list.

        The approval expiry sweep above does not cover it either, whatever its
        docstring used to say: it iterates expired approval *rows*, and an
        ``ask_user`` or ``manual_handoff`` park has no approval row to expire.
        Measured on 2026-08-16 with a 60s grace: five such runs were still
        ``waiting_for_user`` three minutes later.

        Runs after the approval sweep and before any schedule fires. That order
        matters twice: an approval past its deadline gets to end as its own
        expiry (which claims the run, so this sweep skips it), and anything this
        sweep settles is already settled by the time ``_blocking_run_for_task``
        reports on it — so one tick can never abandon a run here and abandon or
        fire over it again there.

        Best-effort, like the sweeps around it: a failure here must not cost
        the tick its due schedules.
        """
        try:
            from agent.approval_resume import is_settling_run

            grace = _stall_grace_seconds()
            for run in await asyncio.to_thread(_parked_runs):
                if is_settling_run(run.get("id")):
                    # Already being ended — by the approval sweep a moment ago,
                    # or by a decision the user is making right now.
                    continue
                elapsed = _waiting_seconds(run)
                if elapsed is not None and elapsed < grace:
                    continue
                logger.warning(
                    "scheduler: abandoning run %s — waiting on a human for "
                    "longer than %ss",
                    run.get("id"),
                    grace,
                )
                await _abandon_stalled_run(run)
        except Exception:
            logger.exception("scheduler: stalled park sweep failed")

    async def _sweep_expired_approvals(self) -> None:
        """Settle approvals whose deadline passed, on this same tick.

        An approval nobody answers parks its run in ``waiting_for_user``
        indefinitely, and — because that counts as active above — takes the
        task's schedule down with it. This sweep ends the approval on its own
        deadline, which is what lets a run be settled *as an expired approval*
        (row recorded expired, audited, denied) rather than merely abandoned.

        Its scope is exactly the approval rows: a park with no approval behind
        it is invisible here, which is why :meth:`_sweep_stalled_parks` exists.
        This docstring used to claim it covered "a run started by hand, or one
        whose schedule is disabled" in general — it never did, and an
        ``ask_user`` park on such a run sat forever. Between the two sweeps the
        coverage is now whole, and both end a run the same way, through
        :func:`agent.approval_resume.abandon_waiting_run` / the deny path, so
        which one gets there first changes the recorded reason but not the
        outcome.

        Every tick rather than on its own cadence: it is one indexed query
        against a table that is nearly always empty of expired rows, and the
        cost of being late is a run that stays stuck. Best-effort — an approval
        sweep failure must never cost a scheduled run.
        """
        try:
            from agent.approval_resume import sweep_expired_approvals

            await sweep_expired_approvals()
        except Exception:
            logger.exception("scheduler: approval expiry sweep failed")

    async def _sweep_cli_agents_if_due(self) -> None:
        """Let the CLI agent sweep piggyback on this loop, if it is due.

        Deliberately not a second background task: this process already owns
        exactly one periodic timer, and a second one would mean two things to
        start, two to stop, and two ways for the lifespan to leak a task. The
        sweep's own cadence (hours, see :mod:`agent.cli_agent_sweep`) is far
        coarser than this tick, so it decides for itself whether it is due
        from a persisted timestamp and almost always answers no.

        Strictly best-effort and strictly last: firing due schedules is this
        loop's job, and a filesystem walk failing must never cost a scheduled
        run.
        """
        try:
            from agent.cli_agent_sweep import maybe_run_due_cli_agent_sweep

            await asyncio.to_thread(maybe_run_due_cli_agent_sweep)
        except Exception:
            logger.exception("scheduler: cli agent sweep failed")


_scheduler: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
