"""Which policy operations a workflow step type actually consults at runtime.

The dashboard's readiness rail needs to answer "will this agent's scheduled
run stall waiting for an approval nobody can see?" before it ever fires. That
question used to be answered by a hand-written map in
``dashboard/templates/agents.html`` (``OPERATION_FOR_STEP``) that guessed at
which step types need a standing policy rule. It guessed wrong in the way
that mattered most: it never mapped ``llm`` steps to anything, so a flow
whose only actually-gated operation was the model's own tool use looked
"ready" right up until the first scheduled run parked forever.

This module replaces the guess with a mapping read off the two runtime paths
that can actually gate a workflow step's execution:

1. **`llm` steps** run through ``_execute_llm_workflow_step`` /
   ``stream_claude_turn`` (``agent/task_orchestrator.py``), which drives the
   same tool-permission control loop
   ``chat_stream_service._handle_control_request`` uses for interactive chat
   (see ``chat_stream_service.py`` around line 654, gated on ``agent_run_id``
   being set on the websocket — that attribute is what marks a turn as an
   unattended agent run rather than a live chat session). Every tool call the
   model makes during that turn is mapped to a policy operation by
   ``_approval_operation_for_tool``, and that function can return any of
   ``LLM_TOOL_APPROVAL_OPERATIONS`` depending on which tool the model happens
   to reach for. Nothing on this side of the model call can predict which of
   the four it will actually use for a given run — an ``llm`` step whose
   prompt is "summarize the last build log" might only ever trigger
   ``provider.tool``, or it might also shell out, write a file, or commit.
   Rather than pretend to that precision, an ``llm`` step is reported as
   requiring the *entire* ``LLM_TOOL_APPROVAL_OPERATIONS`` set: a deliberate,
   documented superset, honest about not knowing the subset in advance
   rather than a claim that every ``llm`` step definitely needs all four.

2. **Every other current workflow step type** — ``shell``, ``browser_action``,
   ``device_action``, ``android_action``, ``mobile_action``, ``app_action``,
   ``notify``, ``condition``, ``manual_handoff``, ``mcp_tool`` and
   ``approval_gate`` — is executed by its own dedicated helper in
   ``agent/task_orchestrator.py`` (``_execute_shell_workflow_step``,
   ``_execute_browser_action_workflow_step``,
   ``_execute_app_action_workflow_step``, ``_execute_notify_workflow_step``,
   or the ``_wait_for_user_step`` branch for the three that always pause),
   and *none of them call* ``evaluate_direct_action_gate`` *or consult any
   standing policy rule*. ``_execute_shell_workflow_step`` runs a script that
   was already vetted when it was registered ("No model, no tokens, no
   approval — the vetting happened when the script was registered"); the
   app/browser-action helpers call ``execute_app_actions`` /
   ``execute_browser_actions`` directly with no gate call anywhere in either
   module. ``device.control`` is a real operation — but only for the
   *direct-action* API surface a phone hits interactively outside a scheduled
   run (see ``routes/policies.py``); it is never what a workflow step's own
   execution consults. Reporting it as "required" for ``app_action`` /
   ``device_action`` would repeat the exact overclaim this module exists to
   remove: a requirement the runtime does not actually enforce.

If a future step type introduces a real gate, or if a step type currently
routed through the ``llm`` fallthrough gets its own dedicated (ungated)
executor, this mapping must change to match — ``tests/
test_required_operations_for_flow.py`` reads both this module's source and
``task_orchestrator.py``'s and fails if they drift apart.
"""

from typing import Any

from chat.chat_stream_service import LLM_TOOL_APPROVAL_OPERATIONS

# Step types with a dedicated executor in agent/task_orchestrator.py's
# `_execute_single_workflow_task_step` that never calls
# `evaluate_direct_action_gate` or otherwise consults a standing policy rule.
# Anything NOT in this set — including "llm" and any future/unrecognised step
# type — falls through to `_execute_llm_workflow_step` in the real runtime
# (task_orchestrator.py's own `_execute_single_workflow_task_step` ends with
# an unconditional call to it after every named branch above), so treating
# the fallthrough as "needs the llm tool-permission set" mirrors the code
# exactly rather than assuming a step type we have never seen.
UNGATED_WORKFLOW_STEP_TYPES: frozenset[str] = frozenset(
    {
        "shell",
        "browser_action",
        "device_action",
        "android_action",
        "mobile_action",
        "app_action",
        "notify",
        "condition",
        "manual_handoff",
        "mcp_tool",
        "approval_gate",
    }
)


def required_operations_for_step_type(step_type: Any) -> tuple[str, ...]:
    """Policy operations a single workflow step type will consult at runtime."""
    normalized = str(step_type or "").strip().lower()
    if normalized in UNGATED_WORKFLOW_STEP_TYPES:
        return ()
    return LLM_TOOL_APPROVAL_OPERATIONS


def required_operations_for_flow(flow: list[dict[str, Any]] | None) -> list[str]:
    """Every operation an agent's flow will need a standing rule for.

    Deduped and ordered by first appearance in the flow, so the result is
    stable for display without the caller needing to sort it.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for step in flow or []:
        step_type = step.get("type") if isinstance(step, dict) else None
        for operation in required_operations_for_step_type(step_type):
            if operation not in seen:
                seen.add(operation)
                ordered.append(operation)
    return ordered


def step_type_operation_catalog() -> dict[str, list[str]]:
    """Map of step type -> required operations, for a client to cache once
    and apply locally over each agent's flow instead of re-deriving it (or,
    worse, hand-copying it) per render.

    Includes ``llm`` explicitly alongside every named ungated type so a
    caller never needs its own fallthrough logic — the map already has an
    entry for every step type the workflow step schema can produce.
    """
    step_types = sorted(UNGATED_WORKFLOW_STEP_TYPES | {"llm"})
    return {step_type: list(required_operations_for_step_type(step_type)) for step_type in step_types}


__all__ = [
    "UNGATED_WORKFLOW_STEP_TYPES",
    "required_operations_for_step_type",
    "required_operations_for_flow",
    "step_type_operation_catalog",
]
