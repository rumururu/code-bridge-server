"""Pure step-ordering cursor for workflow runs.

Every "which step runs next" decision the workflow loop makes lives here, in
one interface, with **no store access and no side effects**: the cursor reads
the steps list and each step's policy dicts (``on_success`` / ``on_failure``)
and answers *where execution goes* — nothing else. Recording what happened
(event appends, step status updates, run finishing, parking on a person) stays
with the orchestrator, because those writes are the run's observable history
and must keep happening exactly where and when they do today.

That split — the cursor answers "where", the caller records — is the whole
point of this module. T-B-06 extracted it; T-B-07 makes this interface the
kernel delegation point: a kernel-backed worker can replace :class:`StepCursor`
wholesale and the orchestrator keeps its recording behavior untouched, because
ordering decisions no longer live inline in the loop. Had even one decision
stayed inline, delegation would fork the routing into two half-authorities.

The decision logic here is a verbatim port of what used to be inline in
``task_orchestrator._drive_workflow_steps`` and its policy helpers
(``_apply_workflow_success_policy`` / ``_apply_workflow_failure_policy`` /
``_apply_terminal_or_branch_policy``). Behavior is intentionally identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SuccessRoute:
    """Where execution goes after a step succeeds.

    ``kind`` is one of:

    * ``"advance"`` — run the step at ``next_index`` (the default: next step).
    * ``"goto"`` — jump to ``next_index`` (resolved from ``target_step_id``);
      the caller records the ``task.step.goto`` event.
    * ``"end"`` — the workflow finishes here; the caller closes the run from
      its steps (failed if any step failed along the way).
    * ``"abort"`` — the policy itself is broken (goto target missing); the
      caller fails the run with ``error_message``.
    """

    kind: str
    next_index: int | None = None
    target_step_id: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class FailureRoute:
    """Where execution goes after a step fails.

    ``kind`` is one of:

    * ``"retry"`` — re-queue the failed step (``next_index`` is the failed
      index); ``attempt``/``max_attempts`` tell the caller what to record.
    * ``"continue"`` — the step stays failed, run the step at ``next_index``.
    * ``"goto"`` — jump to ``next_index`` (resolved from ``target_step_id``).
    * ``"park"`` — wait for a person; ``park_reason`` is the policy type
      (``ask_user`` / ``manual_handoff``) and ``park_prompt`` its prompt.
    * ``"abort"`` — the run ends failed. ``error_message`` set means the
      *policy* is broken and its message replaces the step's own error;
      unset means the caller reports the step's error as-is.
    """

    kind: str
    next_index: int | None = None
    target_step_id: str | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    park_reason: str | None = None
    park_prompt: str | None = None
    error_message: str | None = None


def workflow_step_index(steps: list[dict[str, Any]], workflow_step_id: str) -> int | None:
    """Index of the step whose input carries this workflow step id, if any."""
    for index, step in enumerate(steps):
        step_input = step.get("input")
        if not isinstance(step_input, dict):
            continue
        if step_input.get("workflow_step_id") == workflow_step_id:
            return index
    return None


class StepCursor:
    """Cursor over a workflow run's steps: position, budget, and routing.

    Holds the current index and the transition budget, and interprets the
    success/failure policies into routes. Pure by contract: methods take the
    (possibly refreshed) steps list as input and return decisions; they never
    touch a store, append an event, or mutate a step.
    """

    def __init__(self, steps: list[dict[str, Any]], *, max_transitions: int | None = None) -> None:
        self.index = 0
        self.transitions = 0
        # Same bound the loop always used: generous enough for retries and
        # gotos, small enough that a routing cycle cannot spin forever.
        self.max_transitions = (
            max_transitions if max_transitions is not None else max(10, len(steps) * 10)
        )

    def begin_transition(self) -> bool:
        """Count one loop transition; ``False`` when the budget is exhausted."""
        self.transitions += 1
        return self.transitions <= self.max_transitions

    @staticmethod
    def should_skip(step: dict[str, Any]) -> bool:
        """Steps the loop walks past without running.

        Already-completed steps, and rows that are not workflow-backed (no
        ``workflow_step_id`` on their input), are not work — the cursor just
        moves over them.
        """
        if step.get("status") == "completed":
            return True
        step_input = step.get("input")
        if not isinstance(step_input, dict) or not step_input.get("workflow_step_id"):
            return True
        return False

    def advance_on_success(
        self, steps: list[dict[str, Any]], completed_index: int
    ) -> SuccessRoute:
        """Interpret ``on_success`` for the step that just completed.

        Default is the next step, which is what every workflow did before the
        policy existed. ``end`` is what lets a diagnosis step be reachable only
        through ``on_failure: goto_step`` instead of running on every clean
        night.
        """
        step = steps[completed_index]
        step_input = step.get("input") if isinstance(step.get("input"), dict) else {}
        policy = step_input.get("on_success")
        if not isinstance(policy, dict):
            return SuccessRoute(kind="advance", next_index=completed_index + 1)
        policy_type = str(policy.get("type") or "continue")

        if policy_type == "goto_step":
            target = policy.get("target_step_id") or policy.get("step_id") or policy.get("target")
            target_index = (
                workflow_step_index(steps, target) if isinstance(target, str) else None
            )
            if target_index is None:
                return SuccessRoute(
                    kind="abort",
                    error_message=f"on_success goto_step target not found: {target}",
                )
            return SuccessRoute(kind="goto", next_index=target_index, target_step_id=target)

        if policy_type == "end":
            return SuccessRoute(kind="end")

        return SuccessRoute(kind="advance", next_index=completed_index + 1)

    def route_on_failure(
        self, steps: list[dict[str, Any]], failed_index: int
    ) -> FailureRoute:
        """Interpret ``on_failure`` for the step that just failed.

        ``retry`` counts attempts out of the step input's ``retry_state`` and
        falls through to its ``then`` policy once exhausted; everything else
        resolves through the terminal/branch vocabulary (continue / goto_step /
        ask_user / manual_handoff / abort). No policy means abort.
        """
        failed_step = steps[failed_index]
        step_input = (
            failed_step.get("input") if isinstance(failed_step.get("input"), dict) else {}
        )
        policy = step_input.get("on_failure")
        if not isinstance(policy, dict):
            policy = {"type": "abort"}
        policy_type = str(policy.get("type") or "abort")

        if policy_type == "retry":
            retry_state = step_input.get("retry_state")
            if not isinstance(retry_state, dict):
                retry_state = {}
            attempts = int(retry_state.get("attempts") or 0)
            max_attempts = int(policy.get("max_attempts") or policy.get("max_retries") or 1)
            if attempts < max_attempts:
                return FailureRoute(
                    kind="retry",
                    next_index=failed_index,
                    attempt=attempts + 1,
                    max_attempts=max_attempts,
                )
            then_policy = policy.get("then")
            if isinstance(then_policy, dict):
                return self._terminal_or_branch(steps, failed_index, then_policy)
            return FailureRoute(kind="abort")

        return self._terminal_or_branch(steps, failed_index, policy)

    @staticmethod
    def _terminal_or_branch(
        steps: list[dict[str, Any]], failed_index: int, policy: dict[str, Any]
    ) -> FailureRoute:
        policy_type = str(policy.get("type") or "abort")
        if policy_type == "goto":
            policy_type = "goto_step"

        if policy_type == "continue":
            # The step stays failed — this is not a pass. The run carries on so
            # that work which does not depend on it still happens.
            return FailureRoute(kind="continue", next_index=failed_index + 1)

        if policy_type == "goto_step":
            target = policy.get("target_step_id") or policy.get("step_id") or policy.get("target")
            if not isinstance(target, str) or not target:
                return FailureRoute(
                    kind="abort",
                    error_message="goto_step failure policy is missing target_step_id.",
                )
            target_index = workflow_step_index(steps, target)
            if target_index is None:
                return FailureRoute(
                    kind="abort",
                    error_message=f"goto_step target not found: {target}",
                )
            return FailureRoute(kind="goto", next_index=target_index, target_step_id=target)

        if policy_type in {"ask_user", "manual_handoff"}:
            prompt = policy.get("prompt")
            return FailureRoute(
                kind="park",
                park_reason=policy_type,
                park_prompt=prompt if isinstance(prompt, str) else None,
            )

        return FailureRoute(kind="abort")
