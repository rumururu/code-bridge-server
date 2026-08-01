"""Close out runs that a previous process left mid-flight.

Orchestration lives in the server process: a run marked ``running`` is being
driven by a coroutine in *this* interpreter. So when the server exits — a
restart, a crash, a kill — every in-flight run's row is frozen at whatever it
said, and nobody will ever finish it.

That is not just untidy. ``skip_if_active`` treats a progressing run as work in
progress, and unlike a run waiting on a human it has no grace period, so the
schedule that owns it skips every fire from then on. One restart at the wrong
moment silently retires a schedule for good.

Anything still progressing when the process starts is therefore stale by
definition — nothing has had a chance to run yet — and is failed here with a
reason that says so, rather than left to look like work that is still happening.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.agent_store import get_agent_store

logger = logging.getLogger(__name__)

# Mirrors the scheduler's view of "this is doing work".
INTERRUPTED_RUN_STATUSES = ("queued", "starting", "running")
INTERRUPTED_STEP_STATUSES = ("queued", "running")

_REASON = "Interrupted: the server stopped while this run was in progress."


def reconcile_interrupted_runs() -> list[dict[str, Any]]:
    """Fail runs and steps left mid-flight by a previous process.

    Returns the runs that were closed, for logging and tests.
    """
    try:
        store = get_agent_store()
    except Exception:
        logger.exception("run reconciliation: agent store unavailable")
        return []

    try:
        stale: list[dict[str, Any]] = []
        for status in INTERRUPTED_RUN_STATUSES:
            stale.extend(store.list_runs(status=status, limit=200))
    except Exception:
        logger.exception("run reconciliation: could not list runs")
        return []

    closed: list[dict[str, Any]] = []
    for run in stale:
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id:
            continue
        try:
            store.update_run_status(run_id, "failed")
            _fail_open_steps(store, run)
            store.append_event(
                run_id=run_id,
                event_type="task.execution.failed",
                app_event={
                    "task_id": run.get("task_id"),
                    "result": {},
                    "error": {"message": _REASON},
                },
            )
            closed.append(run)
        except Exception:
            logger.exception("run reconciliation: could not close run %s", run_id)

    if closed:
        logger.warning(
            "run reconciliation: closed %d run(s) interrupted by a previous "
            "shutdown; their schedules were being skipped",
            len(closed),
        )
    return closed


def _fail_open_steps(store: Any, run: dict[str, Any]) -> None:
    """Mark the run's unfinished steps failed too.

    A step left ``running`` keeps the run looking alive to anything that reads
    steps rather than the run row, and leaves the results panel showing a step
    that never resolves.
    """
    task_id = run.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return
    for step in store.list_task_steps(task_id):
        if step.get("status") not in INTERRUPTED_STEP_STATUSES:
            continue
        output = dict(step.get("output") or {})
        output["error"] = {"message": _REASON}
        store.update_task_step(step["id"], {"status": "failed", "output": output})


__all__ = ["reconcile_interrupted_runs", "INTERRUPTED_RUN_STATUSES"]
