"""Turn an approval decision back into a running agent run.

Deciding an approval used to be a dead end: ``approvals/approval_service.py``
recorded the decision and wrote an audit event, and nothing anywhere read it.
A run that had parked waiting for that approval stayed parked forever, so the
user tapped "approve" on their phone and watched nothing happen.

This module is the missing consumer. The decision endpoints call
:func:`maybe_resume_run_for_decision` right after the decision is recorded; it
checks that this approval really is the one a run is parked on, and if so
spawns the orchestrator again with the decision attached so the parked
provider turn can be answered.

It deliberately does *not* live in ``approval_service``: that module is the
low-level, synchronous approval store wrapper, and importing the orchestrator
from it would make an import cycle (orchestrator -> chat_stream_service ->
approval_service). The route layer is where the two sides can meet.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from approvals.approval_service import expire_approval, expire_stale_approvals

from .agent_store import get_agent_store
from .task_orchestrator import (
    abandon_run_waiting_for_a_person,
    execute_task_orchestration,
    resume_task_orchestration,
)

logger = logging.getLogger(__name__)

#: What an expiry hands the orchestrator as its decision. It is inside
#: ``task_orchestrator._APPROVAL_DENY_DECISIONS``: an approval nobody answered
#: in time is a refusal, and the parked turn is answered with a denial rather
#: than left holding its callback forever.
EXPIRED_DECISION = "expired"

#: The checkpoint reason an llm step writes when it parks on a tool approval
#: (``task_orchestrator._execute_llm_workflow_step``). A decision only resumes
#: a run parked for *this* reason — a run waiting on a manual handoff or a
#: captcha is waiting for something else entirely.
APPROVAL_CHECKPOINT_REASON = "approval_required"

# Strong references to spawned resumes. Without this the event loop only holds
# a weak reference and a fire-and-forget orchestrator turn can be collected
# mid-run. Same reasoning (and same shape) as `routes/agents.py`'s
# `_spawn_background`; re-implemented here rather than imported so this module
# does not depend on the route layer.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

#: Runs this module is in the middle of settling, by run id.
#:
#: Every way a parked run can be ended now arrives here — a decision from the
#: app, the expiry sweep, the scheduler giving up — and two of them ride the
#: *same* scheduler tick. Without a claim, an approval expiring at the same
#: moment the schedule checks for stalled runs would be resumed by the sweep
#: and abandoned by the scheduler, driving one run down two paths at once.
#: The claim is held from the moment a settlement starts until the orchestrator
#: turn it spawned is done (which is when the run has finished or parked
#: again), so a second attempt in that window is refused rather than queued.
_SETTLING_RUNS: set[str] = set()


def _claim_settlement(run_id: str) -> bool:
    """Take the right to settle ``run_id``; ``False`` if somebody else has it."""
    if run_id in _SETTLING_RUNS:
        return False
    _SETTLING_RUNS.add(run_id)
    return True


def _release_settlement(run_id: str) -> None:
    _SETTLING_RUNS.discard(run_id)


def is_settling_run(run_id: str | None) -> bool:
    """Is this run being ended right now?

    The scheduler asks before deciding whether a waiting run blocks its
    schedule: a run mid-settlement is neither "waiting on a human" (nobody can
    answer it any more) nor free to be abandoned again.
    """
    return isinstance(run_id, str) and run_id in _SETTLING_RUNS


def _spawn_background(coro: Any, *, run_id: str | None = None) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        _BACKGROUND_TASKS.discard(finished)
        if isinstance(run_id, str) and run_id:
            _release_settlement(run_id)

    task.add_done_callback(_done)


async def maybe_resume_run_for_decision(
    approval: dict[str, Any] | None,
    *,
    decision: str,
) -> dict[str, Any] | None:
    """Resume the run this approval was blocking, if it was blocking one.

    Returns a small record of what was resumed (for the endpoint's response and
    for tests), or ``None`` when nothing matched — an approval that belongs to
    no run, a run that is not parked, a run parked on a *different* approval, or
    a run something else is already settling. Never raises: a decision must
    still be recorded even if the resume cannot be arranged, so failures are
    logged and reported as "not resumed" rather than turned into a 500 on the
    decision endpoint.
    """
    if not isinstance(approval, dict):
        return None
    run_id = approval.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    if not _claim_settlement(run_id):
        logger.info(
            "approval %s decided but run %s is already being settled",
            approval.get("id"),
            run_id,
        )
        return None
    resumed = await _resume_claimed_run(approval, decision=decision)
    if resumed is None:
        # Nothing was spawned, so nothing will release the claim later.
        _release_settlement(run_id)
    return resumed


async def _resume_claimed_run(
    approval: dict[str, Any],
    *,
    decision: str,
) -> dict[str, Any] | None:
    """Body of :func:`maybe_resume_run_for_decision`, claim already held.

    Split out so the abandonment path can reuse it without taking the claim a
    second time (and deadlocking itself out of its own settlement).
    """
    approval_id = approval.get("id")
    run_id = approval.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None

    try:
        store = get_agent_store()
        run = store.get_run(run_id)
        if not isinstance(run, dict):
            return None
        task_id = run.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return None

        checkpoint_context = store.get_task_checkpoint(task_id)
        checkpoint = (
            checkpoint_context.get("checkpoint")
            if isinstance(checkpoint_context, dict)
            else None
        )
        if not isinstance(checkpoint, dict):
            return None
        if checkpoint.get("reason") != APPROVAL_CHECKPOINT_REASON:
            return None

        checkpoint_run = (
            checkpoint_context.get("run") if isinstance(checkpoint_context, dict) else None
        )
        if isinstance(checkpoint_run, dict) and checkpoint_run.get("id") != run_id:
            # The task's active checkpoint belongs to a different run of the
            # same task. Resuming would drive the wrong run.
            return None

        checkpoint_approval_id = checkpoint.get("approval_id")
        if (
            isinstance(checkpoint_approval_id, str)
            and checkpoint_approval_id
            and checkpoint_approval_id != approval_id
        ):
            # Parked on some other approval — deciding this one answers
            # nothing the run is waiting for.
            return None

        try:
            resume = resume_task_orchestration(task_id, permission_decision=decision)
        except ValueError as exc:
            logger.info(
                "approval %s decided but run %s cannot resume: %s",
                approval_id,
                run_id,
                exc,
            )
            return None
        if not isinstance(resume, dict):
            return None
        execution = resume.get("execution")
        if not isinstance(execution, dict):
            return None

        # The park this decision answers is over. Clearing the marker matters
        # for more than tidiness: `_wait_for_user_step` suppresses a repeat
        # notification when the *previous* checkpoint had the same
        # run/step/reason, so leaving a resolved marker in place would silence
        # the push for the next tool the same step stops on.
        task = store.get_task(task_id)
        if isinstance(task, dict):
            metadata = dict(task.get("metadata") or {})
            if metadata.pop("active_checkpoint", None) is not None:
                store.update_task(task_id, {"metadata": metadata})

        _spawn_background(execute_task_orchestration(execution), run_id=run_id)
        logger.info(
            "approval %s decided (%s); resumed run %s task %s",
            approval_id,
            decision,
            run_id,
            task_id,
        )
        return {
            "approval_id": approval_id,
            "run_id": run_id,
            "task_id": task_id,
            "step_id": checkpoint.get("step_id"),
            "decision": decision,
            "resumed": True,
        }
    except Exception:  # noqa: BLE001 - the decision itself must still succeed
        logger.exception(
            "failed to resume run %s after deciding approval %s", run_id, approval_id
        )
        return None


async def sweep_expired_approvals() -> list[dict[str, Any]]:
    """Expire approvals past their deadline and settle the runs they parked.

    Without this an unanswered approval parks its run in ``waiting_for_user``
    for good: the row stays ``pending`` (and is hidden from the pending list
    once ``expires_at`` passes, so nobody can even answer it), the run keeps
    counting as active, and the provider session sits holding an unanswered
    ``can_use_tool`` callback. Expiring is deliberately *not* a quiet cleanup:
    each row is recorded as expired, and the run it blocked is resumed down the
    same deny path a refusal takes, so the model is told the tool call was
    refused and the step's failure policy gets to decide what happens next.

    Runs the scheduler tick calls this from; safe to call at any time. Never
    raises — a failure here must not cost the scheduler its tick.
    """
    try:
        expired = await asyncio.to_thread(expire_stale_approvals)
    except Exception:
        logger.exception("approval expiry sweep failed to expire rows")
        return []

    swept: list[dict[str, Any]] = []
    for approval in expired:
        record: dict[str, Any] = {
            "approval_id": approval.get("id"),
            "run_id": approval.get("run_id"),
            "expires_at": approval.get("expires_at"),
            "resumed": False,
        }
        resumed = await maybe_resume_run_for_decision(
            approval, decision=EXPIRED_DECISION
        )
        if isinstance(resumed, dict):
            record.update(resumed)
        else:
            # Nothing was parked on it (a chat-side approval, or a run that
            # already moved on). The row is still recorded as expired, which is
            # the whole point — it is no longer answerable.
            logger.info(
                "approval %s expired; no parked run to resume", approval.get("id")
            )
        swept.append(record)
    if swept:
        logger.info("approval expiry sweep settled %d approval(s)", len(swept))
    return swept


def _parked_approval_for_run(run_id: str) -> dict[str, Any] | None:
    """The approval a run is parked on right now, or ``None`` if it is not.

    Read from the run's own checkpoint rather than from the approvals table:
    a run can have several approval rows in its history and only one of them is
    the thing it stopped for. When the row itself is gone (a pruned or lost
    approval) the identity from the checkpoint is enough — the deny path is
    driven by the explicit decision, not by re-reading the row.
    """
    try:
        store = get_agent_store()
        run = store.get_run(run_id)
        if not isinstance(run, dict):
            return None
        task_id = run.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return None
        context = store.get_task_checkpoint(task_id)
        checkpoint = context.get("checkpoint") if isinstance(context, dict) else None
        if not isinstance(checkpoint, dict):
            return None
        if checkpoint.get("reason") != APPROVAL_CHECKPOINT_REASON:
            # An `ask_user` or manual-handoff park. There is no approval here,
            # and pretending there is would send a run with nothing to deny
            # down the deny path.
            return None
        checkpoint_run = context.get("run") if isinstance(context, dict) else None
        if isinstance(checkpoint_run, dict) and checkpoint_run.get("id") != run_id:
            return None
        approval_id = checkpoint.get("approval_id")
        if isinstance(approval_id, str) and approval_id:
            from approvals.approval_store import get_approval_store

            row = get_approval_store().get_request(approval_id)
            if isinstance(row, dict):
                return row
        return {"id": approval_id, "run_id": run_id}
    except Exception:
        logger.exception("could not read the approval run %s is parked on", run_id)
        return None


def _expire_pending_approvals_for_run(run_id: str) -> list[str]:
    """Expire every approval still answerable (or overdue) for this run.

    The run is ending, so its prompts are about something that is no longer
    going anywhere; leaving them ``pending`` would keep offering the user a
    button that decides nothing. ``list_pending`` deliberately hides rows past
    their deadline, so the overdue ones are collected separately — those are
    exactly the rows the expiry sweep has not reached yet.
    """
    try:
        from approvals.approval_store import get_approval_store

        store = get_approval_store()
        rows = list(store.list_pending(run_id=run_id))
        rows.extend(
            row for row in store.list_expired_pending() if row.get("run_id") == run_id
        )
    except Exception:
        logger.exception("could not list approvals for run %s", run_id)
        return []

    expired: list[str] = []
    for row in rows:
        approval_id = row.get("id")
        if not isinstance(approval_id, str) or not approval_id or approval_id in expired:
            continue
        try:
            expire_approval(approval_id, trigger="stall")
        except Exception:
            logger.exception("could not expire approval %s", approval_id)
            continue
        expired.append(approval_id)
    return expired


async def abandon_waiting_run(
    run: dict[str, Any],
    *,
    waited_seconds: float | None = None,
) -> dict[str, Any] | None:
    """End a run nobody answered, down whichever path fits what it is parked on.

    The single entry point for "give up on this run". Before this existed the
    scheduler had its own: it expired the run's approvals and wrote
    ``failed`` straight onto the run row. That skipped everything the deny path
    does — the provider session kept holding its unanswered ``can_use_tool``
    callback (the very leak the expiry work set out to close), and the step was
    failed with no record of *what* had been refused, so the app had nothing to
    show but a red run.

    Two shapes, one door:

    * parked on an **approval** — expire the rows, then resume through
      :func:`_resume_claimed_run` with the ``expired`` decision. That settles
      the parked turn with a denial and lets the step's failure policy run, so
      the ending is identical to an approval that timed out on its own.
    * parked on **anything else** (``ask_user``, a manual handoff) — there is
      no approval to deny, so the run is failed *on the record* by
      :func:`~agent.task_orchestrator.abandon_run_waiting_for_a_person`, which
      says why rather than leaving a bare ``failed``.

    Returns a record of what happened, or ``None`` when nothing was done. Never
    raises: abandonment is best-effort cleanup and must not cost a tick.
    """
    if not isinstance(run, dict):
        return None
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        return None
    if not _claim_settlement(run_id):
        return {"run_id": run_id, "abandoned": False, "reason": "already_settling"}

    spawned = False
    try:
        approval = _parked_approval_for_run(run_id)
        if approval is not None:
            _expire_pending_approvals_for_run(run_id)
            resumed = await _resume_claimed_run(approval, decision=EXPIRED_DECISION)
            if isinstance(resumed, dict):
                spawned = True
                logger.warning(
                    "abandoning run %s: unanswered approval settled as a denial",
                    run_id,
                )
                return {
                    "run_id": run_id,
                    "abandoned": True,
                    "path": "deny",
                    **resumed,
                }
            # The resume could not be arranged (no checkpoint to drive, a run
            # that already moved on). Fall through and end it plainly rather
            # than leaving it parked on an approval that is now expired.

        abandoned = await asyncio.to_thread(
            abandon_run_waiting_for_a_person,
            run_id,
            waited_seconds=waited_seconds,
        )
        if abandoned is None:
            return {"run_id": run_id, "abandoned": False, "reason": "no_longer_waiting"}
        logger.warning("abandoning run %s: nobody answered its park", run_id)
        return {"run_id": run_id, "abandoned": True, "path": "abandon", **abandoned}
    except Exception:
        logger.exception("failed to abandon waiting run %s", run_id)
        return None
    finally:
        if not spawned:
            _release_settlement(run_id)
