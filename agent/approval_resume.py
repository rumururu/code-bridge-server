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

from approvals.approval_service import expire_stale_approvals

from .agent_store import get_agent_store
from .task_orchestrator import execute_task_orchestration, resume_task_orchestration

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


def _spawn_background(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def maybe_resume_run_for_decision(
    approval: dict[str, Any] | None,
    *,
    decision: str,
) -> dict[str, Any] | None:
    """Resume the run this approval was blocking, if it was blocking one.

    Returns a small record of what was resumed (for the endpoint's response and
    for tests), or ``None`` when nothing matched — an approval that belongs to
    no run, a run that is not parked, or a run parked on a *different*
    approval. Never raises: a decision must still be recorded even if the
    resume cannot be arranged, so failures are logged and reported as "not
    resumed" rather than turned into a 500 on the decision endpoint.
    """
    if not isinstance(approval, dict):
        return None
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

        _spawn_background(execute_task_orchestration(execution))
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
